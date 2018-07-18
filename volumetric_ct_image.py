"""
Library for representing 3D volumetric CT scan data and manipulating them
by resampling, etc.
"""
from __future__ import print_function
import sys, os
import numpy as np
import scipy.ndimage.interpolation as interpolation
import transforms3d.affines as affine3d
import copy


def map_coords_to_scaled_float(coords, orig_size, new_size):
    """
    maps coordinates relative to the original 3-D image to coordinates corresponding to the 
    re-scaled 3-D image, given the coordinates and the shapes of the original and "new" scaled
    images.  Returns a floating-point coordinate center where the pixel at array coordinates
    (0,0) has its center at (0.5, 0.5).  Take the floor of the return value from this function
    to get back to indices.
    """
    if not all(
        isinstance(arg, (tuple, list, set)) for arg in (coords, orig_size, new_size)
    ):
        raise TypeError(
            "`coords`, `orig_size` and `new_size` must be tuples corresponding to the image shape."
        )
    if not all(len(arg) == len(coords) for arg in (orig_size, new_size)):
        raise TypeError(
            "Number of dimensions in `coords` ({}), `orig_size` ({}), and `new_size` ({}) did not match.".format(
                len(coords), len(orig_size), len(new_size)
            )
        )

    ratio = lambda dim: float(new_size[dim]) / orig_size[dim]
    center = lambda s, dim: s[dim] / 2.0
    offset = lambda dim: (coords[dim] + 0.5) - center(orig_size, dim)
    new_index = lambda dim: (center(new_size, dim) + ratio(dim) * offset(dim))

    return tuple([new_index(dim) for dim in range(len(orig_size))])


def map_coords_to_scaled(coords, orig_size, new_size):
    """
    maps coordinate indices relative to the original 3-D image to indices corresponding to the 
    re-scaled 3-D image, given the coordinate indices and the shapes of the original 
    and "new" scaled images.  Returns integer indices of the voxel that contains the center of 
    the transformed coordinate location.
    """
    return tuple(
        [int(i) for i in map_coords_to_scaled_float(coords, orig_size, new_size)]
    )


class StandardizedScan(object):
    """
    Represents a CT scan standardized to a common cubic voxel size (by default, 1mm x 1mm x 1mm).
    """

    class TransformError(RuntimeError):
        """ 
        Thrown when transform cannot be applied to image.
        Catch as type `StandardizedScan.TransformError`.
        """

        pass

    def __init__(
        self, dicomdir=None, img=None, hdr=None, mm_per_voxel=1.0, orientation="view"
    ):
        """
        Initialize a standardized scan where each voxel is a cube of size `mm_per_voxel`, given
        a path to a directory containing DICOM image files (`dicomdir`) or an image `img` and 
        corresponding header `hdr` (which must contain an attribute `.Affine` containing the affine 
        transformation matrix corresponding to the image scan).
        Initializing from DICOM is preferred; if `dicomdir` is provided, `img` and `hdr` are ignored.

        The scan may be represented in one of two possible orientations:
        
        'view' -  dimension ordering (Z, Y, X) (or slices, columns, rows) -- this is default
                  since it is easier to reason about in a numpy array sense; it puts the "head"
                  at index 0 and increases Z toward the "feet".  X increases toward the patient's
                  left, Y increases toward the patient's posterior.
        
        'dicom' - dimension ordering (X, Y, Z) -- this is the ordering that corresponds to the
                  DICOM header standard; the Z axis increases from "feet" toward "head", with 
                  index 0 at the "feet".  X increases toward the patient's left, Y increases
                  toward the patient's posterior.

        You may also load the scan from an hdf5 file previously saved by the `.save_hd5` method; to
        do so, place the file name in the `img` parameter.

        If `mm_per_voxel` is set to a negative value, the original voxel shapes will be preserved. 
        If `mm_per_voxel` is a tuple, list, or ndarray of length 3, the voxel shapes will be adjusted
        to the specified sizes (and a negative in any dimension keeps the original size for that 
        dimension).
        """
        self.__ready = False
        try:
            self.__mm_per_voxel = float(mm_per_voxel)
        except TypeError:
            self.__mm_per_voxel = np.array(mm_per_voxel, dtype="float")
        self.__orientation = "view" if orientation.lower() != "dicom" else "dicom"
        if dicomdir is not None:
            self.from_dicom_dir(dicomdir)
        elif img is not None:
            if isinstance(img, str):
                if os.path.splitext(img)[1].lower() in [".hd5", ".hdf5", ".hdf", ".h5"]:
                    self.load_hd5(img)
                else:
                    raise RuntimeError(
                        "`img` must be either an image (in numpy array form) or the filename of an HDF5-format file to load."
                    )
            else:
                self.from_img(img, hdr)

    def from_dicom_dir(self, dicomdir):
        """
        (Re)-initialize the StandardizedScan object given a path to a directory containing DICOM files
        corresponding to slices of the volume.  The orientation will be set to the current orientation of
        the StandardizedScan object.
        """
        import load_dicom_dir as ldd

        img, hdr = ldd.load_dicom_dir_dicom_orientation(dicomdir)
        img = img.astype(np.int16)
        self.from_img(img, hdr, orientation="dicom")

    def from_img(self, img, hdr=None, orientation="dicom"):
        """
        (Re)-initialize the StandardizedScan object given an image (NumPy array) and header (similar
        to pydicom header format); the orientation of `img` must be specified as 'dicom' or 'view', 
        with the default being 'dicom' (X,Y,Z).
        The final orientation will be set to match the current orientation of the StandardizedScan
        object (if it differs from `orientation`).
        """
        orientation = "view" if orientation.lower() != "dicom" else "dicom"
        try:
            if hdr.RescaleSlope != 1.0 and hdr.RescaleIntercept != 0:
                import load_dicom_dir as ldd

                img = rescale_image(img, hdr)
                img = img.astype(np.int16)
                hdr.RescaleIntercept = 0
                hdr.RescaleSlope = 1.0
        except:
            pass
        self.__img = img
        self.__hdr = hdr
        self.__orig_shape = self.__img.shape
        self.__resample()
        if orientation != self.__orientation:
            if orientation == "dicom":
                self.__reorient_dicom_to_view()
            else:
                self.__reorient_view_to_dicom()
        self.__ready = True

    @property
    def shape(self):
        """
        Get the shape of the current StandardizedScan image.
        """
        self.__assert_ready()
        return self.__img.shape

    @property
    def original_shape(self):
        """
        Get the shape of the original image this StandardizedScan was created from.
        """
        self.__assert_ready()
        return copy.deepcopy(self.__orig_shape)

    @property
    def img(self):
        """
        Get the current StandardizedScan image (as a NumPy array).
        """
        self.__assert_ready()
        return self.__img

    @img.setter
    def img(self, img):
        """
        Set the image component of the StandardizedScan to a new image volume, which must be
        a NumPy array of the same shape and orientation as the current image.
        """
        if not issubclass(type(img), np.ndarray):
            raise RuntimeError("Values assigned to `img` must be NumPy arrays.")
        if img.shape != self.__img.shape:
            raise RuntimeError(
                "Cannot assign an image of a different shape to a StandardizedScan object.  Expected shape {}, attepted to assign shape {}.".format(
                    self.__img.shape, img.shape
                )
            )
        self.__img = img.copy()

    @property
    def orientation(self):
        """
        Get the current orientation of the StandardizedScan image.
        """
        return self.__orientation

    @orientation.setter
    def orientation(self, orientation):
        """
        Set orientation to one of ["view", "dicom"].
        """
        orientation = "view" if orientation.lower() != "dicom" else "dicom"
        if self.__orientation != orientation:
            if self.__ready:
                if self.__orientation == "view":
                    self.__reorient_view_to_dicom()
                else:
                    self.__reorient_dicom_to_view()
            else:
                self.__orientation = orientation

    def save_hd5(self, filename, create_path=False):
        """
        Saves the StandardizedScan in HDF5 format so that it can be loaded again
        directly.  Info for mapping original coordinates to the standardized scan 
        volume is preserved.
        """
        self.__assert_ready()
        import h5py

        directory = os.path.dirname(filename)
        basename = os.path.basename(filename)

        if basename == "":
            raise RuntimeError("A non-empty filename must be specified for `save_as`.")
        if not os.path.isdir(directory) and not create_path:
            raise RuntimeError(
                'Directory path "{}" does not exist; use `create_path` option to automatically create it.'.format(
                    directory
                )
            )
        if create_path and not os.path.isdir(directory):
            os.makedirs(directory)

        if not os.path.splitext(basename)[1].lower() in [
            ".hd5",
            ".hdf5",
            ".hdf",
            ".h5",
        ]:
            basename = "{}.hd5".format(basename)

        filename = os.path.join(directory, basename)

        fp = h5py.File(filename, "w")
        fp["img"] = self.__img
        fp["mm_per_voxel"] = self.__mm_per_voxel
        fp["original_shape"] = self.__orig_shape
        fp["dicom_orientation"] = 0 if self.__orientation == "view" else 1
        fp.close()

    def load_hd5(self, filename):
        """
        Loads directly from HD5-format produced by the `save_as` method.
        """
        import h5py

        fp = h5py.File(filename, "r")
        self.__img = fp["img"].value
        self.__mm_per_voxel = fp["mm_per_voxel"].value
        self.__orig_shape = tuple(fp["original_shape"].value)
        file_orientation = "view" if fp["dicom_orientation"].value == 0 else "dicom"
        fp.close()
        self.__hdr = None
        if file_orientation != self.__orientation:
            if file_orientation == "view":
                self.__reorient_view_to_dicom()
            else:
                self.__reorient_dicom_to_view()
        self.__ready = True

    def map_original_coordinates_float(self, coords, orientation="view"):
        """
        Returns a set of voxel-centered (floating-point) coordinates corresponding 
        to coordinates from the original image the StandardizedScan was created from.
        Specify the orientation of the original coordinates in "view" or "dicom" format
        (default is "view" (Z,Y,X)). 
        """
        orientation = "view" if orientation.lower() != "dicom" else "dicom"
        if orientation != self.__orientation:
            coords = list(coords)
            coords[self.__zdim(orientation)] = (
                self.shape[self.__zdim()] - coords[self.__zdim(orientation)] - 1
            )
            coords = tuple(reversed(coords))
        return map_coords_to_scaled_float(coords, self.__orig_shape, self.__img.shape)

    def map_original_coordinates(self, coords, orientation="view"):
        """
        Returns integer coordinates corresponding to coordinates from
        the original image the StandardizedScan was created from. 
        Specify the orientation of the original coordinates in "view" or "dicom" format
        (default is "view" (Z,Y,X)). 
        """
        return tuple(
            [int(i) for i in self.map_original_coordinates_float(coords, orientation)]
        )

    def __resample(self):
        """
        Resample image to standard shape; this occurs only at (re)-initialization time
        """
        shape = self.__img.shape
        affine = None
        hdr = self.__hdr
        try:
            affine = hdr.Affine
        except:
            print("header: {}".format(hdr))
            raise self.TransformError(
                "Image header must contain affine transformation information in `.Affine` attribute."
            )
        _, R, Z, S = affine3d.decompose44(hdr.Affine)
        # If any of the __mm_per_voxel items are negative, keep the original zoom at those locations:
        mm_per_voxel = np.atleast_1d(self.__mm_per_voxel).astype("float")
        if len(mm_per_voxel) not in [1, len(Z)]:
            raise RuntimeError(
                "`mm_per_voxel` must be a scalar value or tuple of length {}.".format(
                    len(Z)
                )
            )
        if any(mm_per_voxel < 0):
            if len(mm_per_voxel) == 1:
                mm_per_voxel = Z
            else:
                mm_per_voxel[mm_per_voxel < 0] = Z[mm_per_voxel < 0]
        # If there are shears, bail out (we don't support that yet):
        if S.sum() != 0:
            raise self.TransformError(
                "Image affine includes shear, which is not supported in this version."
            )
        # See if any rotations are necessary (we don't support that yet):
        if np.any(np.eye(R.shape[0]).astype(R.dtype) != R):
            raise self.TransformError(
                "Image affine includes rotation, which is not supported in this version"
            )
        # Now apply scaling
        Z = Z / np.array(mm_per_voxel)
        self.__img = interpolation.zoom(self.__img, Z)

    def __reorient_dicom_to_view(self):
        """
        Change DICOM (X,Y,Z) orientation to "view" (Z,Y,X) orientation with Z axis inverted (head at index 0).
        """
        self.__img = np.transpose(self.__img, (2, 1, 0))  # Move from (X,Y,Z) to (Z,Y,X)
        self.__img = self.__img[::-1]  # Arrange slices so "head" end is at index 0.
        self.__orig_shape = tuple(
            [self.__orig_shape[2], self.__orig_shape[1], self.__orig_shape[0]]
        )
        self.__orientation = "view"

    def __reorient_view_to_dicom(self):
        """
        Change "view" (Z,Y,X) orientation to "DICOM" (X,Y,Z) orientation with Z axis increasing from feet toward head (feet at index 0).
        """
        self.__img = self.__img[::-1]  # Arrange slices so "feet" end is at index 0.
        self.__img = np.transpose(self.__img, (2, 1, 0))  # Move from (Z,Y,X) to (X,Y,Z)
        self.__orig_shape = tuple(
            [self.__orig_shape[2], self.__orig_shape[1], self.__orig_shape[0]]
        )
        self.__orientation = "dicom"

    def __assert_ready(self):
        """
        Checks that the image has been properly initialized and raises an exception if it has not.
        """
        if not self.__ready:
            raise RuntimeError("StandardizedScan object has not been initialized.")

    def __zdim(self, orientation=None):
        """
        Get the index of the dimension that represents the 'Z' axis in a given orientation; 
        if orientation is omitted, the current 'Z' dimension for this StandardizedScan is 
        returned.
        """
        if orientation is None:
            orientation = self.__orientation
        orientation = "view" if orientation.lower() != "dicom" else "dicom"
        return 0 if orientation == "view" else 2


if __name__ == "__main__":
    print("This script is not designed to be executed directly.")
    sys.exit(1)
