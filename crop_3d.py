"""
Tools for cropping 3D image volumes (represented as Numpy arrays),
and for iterating arrays of these cropped volumes (with lazy cropping).
"""

import numpy as np


class AutoCrop(object):
    def __init__(self, images, shape):
        if isinstance(images, (list, tuple)):
            images = np.array(images)
        if len(images.shape) < 4:
            images = np.array([images])
        self.__needs_crop = True
        self.__len = len(images)
        self.__images = images
        self.__position = 0
        self.__out_shape = shape
        sz, sy, sx = shape
        self.__shape = (images.shape[0], sz, sy, sx)
        self.__orig_shape = images.shape
        n_frames, mz, my, mx = self.__orig_shape
        self.__n_frames = n_frames
        self.__dtype = images[0].dtype
        if sz == mz and sy == my and sx == mx:
            self.__needs_crop = False

    @property
    def shape(self):
        return self.__shape

    def next(self):
        return self.__next__()

    def reset(self):
        self.__position = 0

    def cropped(self):
        return self.__crop(self.__images)

    def __iter__(self):
        return self

    def __crop_many(self, imgs):
        results = []
        for slice in imgs:
            results.append(self.__crop_one(slice))
        return np.array(results, self.__dtype)

    def __crop_one(self, img):
        result = None
        if self.__needs_crop:
            result = crop_centered(img, self.__out_shape)
        else:
            result = img
        return result

    def __crop(self, img):
        result = None
        if len(img.shape) > 3:
            result = self.__crop_many(img)
        else:
            result = self.__crop_one(img)
        return result

    def __next__(self):
        position = self.__position
        self.__position += 1
        if position >= self.__len:
            self.reset()
            raise StopIteration()
        return self.__crop(self.__images[position])

    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        return self.__crop(self.__images[index])

    def __copy__(self):
        return AutoCrop(self.__images.copy(), self.__out_shape)

    def __deepcopy__(self):
        return self.__copy__()


# ---------------------------------------------------------------------


def crop_centered(image, shape, pad_value=None):
    """
    Crop a single image to `shape`; `shape` may be larger than
    the original image, in which case the border is padded with
    the minimum image value or `padding_value` if it is supplied.
    The cropped image is centered on the original image.
    """
    center = tuple(i // 2 for i in image.shape)
    if image.shape != shape:
        image = crop_centered_at_point(image, center, shape, pad_value)
    return image


def crop_pair_centered(image_1, image_2, shape=None):
    """
    Given a pair of images, crop them both to `shape`.  If 
    `shape` is omitted, the shape will be the same as the 
    smallest image.
    """
    if shape == None:
        shape = image_1.shape
        if np.array(image_2.shape).prod() < np.array(image_1.shape).prod():
            shape = image_2.shape

    if (image_1.shape != shape) or (image_2.shape != shape):
        left = crop_centered(image_1, shape)
        right = crop_centered(image_2, shape)
    else:
        left = image_1.copy()
        right = image_2.copy()

    return (left, right)


def crop_centered_at_point(image, center, shape, pad_value=None):
    """
    Crop a single image to `shape`, centering the crop around a 
    point `center`; `shape` may be larger than
    the available image volume, in which case the border is padded 
    with the minimum image value or `padding_value` if it is supplied.
    """
    image = np.array(image).copy()
    pad_value = image.min() if pad_value is None else pad_value
    cz, cy, cx = None, None, None
    oz, oy, ox = None, None, None
    if len(shape) == 3:
        oz, oy, ox = shape
    else:
        oy, ox = shape
        oz = 1

    if len(image.shape) == 3:
        cz, cy, cx = center
    else:
        cy, cx = center
        cz = 0
        image = np.array([image])

    iz, iy, ix = image.shape
    icz, icy, icx = iz // 2, iy // 2, ix // 2
    ocz, ocy, ocx = oz // 2, oy // 2, ox // 2

    ix_0 = max(cx - ocx, 0)
    iy_0 = max(cy - ocy, 0)
    iz_0 = max(cz - ocz, 0)
    ox_0 = max(ocx - cx, 0)
    oy_0 = max(ocy - cy, 0)
    oz_0 = max(ocz - cz, 0)
    wx = min(ix - ix_0, ox - ox_0)
    wy = min(iy - iy_0, oy - oy_0)
    wz = min(iz - iz_0, oz - oz_0)

    cropped = np.ones((oz, oy, ox)).astype(image.dtype) * pad_value

    cropped[oz_0 : oz_0 + wz, oy_0 : oy_0 + wy, ox_0 : ox_0 + wx] = image[
        iz_0 : iz_0 + wz, iy_0 : iy_0 + wy, ix_0 : ix_0 + wx
    ]
    image = cropped
    if len(shape) == 2:
        image = image[0]
    return image


def crop_pair_centered_at_point(image_1, image_2, center, center_2=None, shape=None):
    """
    Given a pair of images, crop them both so that they are centered
    around the point `center` and sized according to `shape`.  If 
    `shape` is omitted, the shape will be the same as the 
    smallest image.
    If the two image centers should be different, the first is supplied in
    `center` and the second in `center_2`; otherwise, `center` will be used
    for both images and `center_2` need not be given.
    """
    if center_2 is None:
        center_2 = center
    if shape is None:
        shape = image_1.shape
        if np.array(image_2.shape).prod() < np.array(image_1.shape).prod():
            shape = image_2.shape

    left = crop_centered_at_point(image_1, center, shape)
    right = crop_centered_at_point(image_2, center_2, shape)

    return (left, right)
