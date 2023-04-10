"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""

from typing import List

import numpy as np

import matplotlib.pyplot as plt

import cv2

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    my_id = np.int_(323825059)
    return my_id


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    image = cv2.imread(filename)
    while True:
        if representation == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif representation == 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            representation = int(input("Got wrong mode, please renter - 1 or 2"))
            continue
        break
    arr_im = np.array(image).astype(np.float64)
    arr_norm = normalize(arr_im)
    return arr_norm


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    im_arr = imReadAndConvert(filename, representation)
    plt.imshow(im_arr)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    flatten = imgRGB.transpose((2, 0, 1)).reshape(3, -1)  # shaping into a multipliable form
    from_RGB_to_YIQ = np.array([[0.299, 0.587, 0.114],  # the matrix that in mul transforms form RGB to YIQ
                                [0.596, -0.275, -0.321],
                                [0.212, -0.523, 0.311]])
    mul = np.matmul(from_RGB_to_YIQ, flatten)
    yiq_img = mul.transpose((1, 0)).reshape(imgRGB.shape)  # shaping back
    return yiq_img


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    flatten = imgYIQ.transpose((2, 0, 1)).reshape(3, -1)
    from_RGB_to_YIQ = np.array([[0.299, 0.587, 0.114],
                                [0.596, -0.275, -0.321],
                                [0.212, -0.523, 0.311]])
    from_YIQ_to_RGB = np.linalg.inv(from_RGB_to_YIQ)  # the inverse matrix - in mul transforms form YIQ to RGB
    mul = np.matmul(from_YIQ_to_RGB, flatten)
    rgb_img = mul.transpose((1, 0)).reshape(imgYIQ.shape)
    return rgb_img


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original Histogram
    :return (imgEq,histOrg,histEQ)
    """
    arr1, is_rgb = image_type_check(imgOrig)

    not_norm_arr = stretch_back(arr1)  # changes values of image to be in range (0, 255)
    histogram, bin_edges = np.histogram(not_norm_arr, bins=256, range=(0, 256))
    cum_sum = np.cumsum(histogram)

    # lookup table
    lut = np.ceil((cum_sum * 255) / (arr1.shape[0] * arr1.shape[1])).astype(int)

    new_arr = np.take(lut, not_norm_arr.astype(int))  # where intensity is i in not_norm_arr - assigns lut[i]
    new_histogram, new_bin_edges = np.histogram(new_arr, bins=256, range=(0, 256))

    if is_rgb:
        tmp = transformRGB2YIQ(stretch_back(imgOrig))
        tmp[:, :, 0] = new_arr
        new_arr = transformYIQ2RGB(tmp)

    new_arr = normalize(new_arr)
    return new_arr, histogram, new_histogram


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
    Quantized an image in to **nQuant** colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i],List[error_i])
    """
    if nQuant >= 256:
        return imOrig

    arr1, is_rgb = image_type_check(imOrig)

    qImage_i = []
    error_i = []

    # change values of image to be in range (0, 255)
    not_norm_arr = stretch_back(arr1)

    # histogram building
    histogram, bin_edges = np.histogram(not_norm_arr, bins=256, range=(0, 256))
    histogram = histogram.reshape(256, 1)
    bin_edges = bin_edges[:256].reshape(256, 1)

    # if the number asked fr is not reachable
    if np.count_nonzero(histogram) <= nQuant:
        return imOrig

    num_of_pixels = imOrig.shape[0] * imOrig.shape[1]
    np_initial_seg = num_of_pixels / nQuant
    z = initial_borders(histogram, np_initial_seg, nQuant + 1)

    # changing the borders and values loop
    for i in range(0, nIter):
        q = getWeightedMean(bin_edges, histogram, z)
        z = getNewBorders(q)
        new_image = apply_quant(q, z, np.copy(not_norm_arr))

        mse = calc_mse(not_norm_arr, new_image, num_of_pixels)
        error_i.append(mse)

        if is_rgb:
            tmp = transformRGB2YIQ(stretch_back(imOrig))
            tmp[:, :, 0] = new_image
            new_image = transformYIQ2RGB(tmp)

        qImage_i.append(normalize(new_image))

        # checking if converges
        if mse == 0 or (len(error_i) > 2 and mse in error_i):
            if mse in error_i:
                if len(error_i) > 2 and mse in error_i:
                    indices = np.where(error_i == mse)[0]
                    for idx in indices:
                        if np.array_equal(qImage_i[idx], new_image):
                            break
            else:
                break

    return qImage_i, error_i


def image_type_check(imOrig: np.ndarray) -> (np.ndarray, bool):
    """
    The function checks is the image is RGB or gray scale based on the shape of the image
    :param imOrig: the image the function checks
    :return: the image if it's a gray scale, the Y dimension of the transformation to YIQ if the picture is RGB,
    and a boolean indicates if the image was RGB or not
    """
    is_rgb = False
    if len(imOrig.shape) == 3 and imOrig.shape[2] == 3:
        # means that the image has a third dimension - RGB representation
        arr1 = transformRGB2YIQ(imOrig)[:, :, 0]
        is_rgb = True
    elif len(imOrig.shape) == 2:
        arr1 = imOrig
    else:
        raise Exception("The array got is not a proper RGB/GRAY_SCALE image!")
    return arr1, is_rgb


def getWeightedMean(intensity: np.ndarray, pixels_in_intensity: np.ndarray, borders: np.ndarray) -> np.ndarray:
    """
    The function calculates the weighted mean of each one of the given segments (given by segments borders)
    :param intensity: an array with the value range of pixels in the image
    :param pixels_in_intensity: array with num of pixels at each intensity
    :param borders: the borders between segments in the image
    :return: an array with the mean value range of each of the segments in the array
    """
    means = np.empty(len(borders) - 1)
    borders = np.ceil(borders).astype(int)
    for i in range(0, len(borders) - 1):
        val = np.dot(intensity[borders[i]:borders[i + 1]].T, pixels_in_intensity[borders[i]:borders[i + 1]])
        n_pixels = pixels_in_intensity[borders[i]:borders[i + 1]].sum()
        means[i] = val / n_pixels
    return means


def getNewBorders(seg_values: np.ndarray) -> np.ndarray:
    """
    The function calculates the new borders by the mean of the two weighted mean values given
    :param seg_values: an array of the mean segment values in each segment at the original image
    :return: an array with the new borders values
    """
    shift = np.roll(seg_values, 1)
    new_borders = (shift + seg_values) / 2
    new_borders[0] = 0
    new_borders = np.append(new_borders, 255)
    return new_borders


def apply_quant(values: np.ndarray, borders: np.ndarray, org_im: np.ndarray) -> np.ndarray:
    """
    The function applies a given quantization
    :param values: an array of the values each segment maps to
    :param borders: an array of the borders to each segment
    :param org_im: an array of the original image
    :return: the original image after the color changing has been applied to it
    """
    for i in range(0, len(values)):
        org_im[(org_im >= borders[i]) & (org_im <= borders[i + 1])] = values[i]
    return org_im


def calc_mse(orgarr: np.ndarray, quantarr: np.ndarray, n_pixels: int) -> int:
    """
    The function calculates the mse between the original image arr to the applied quantization one
    :param orgarr: array of the original image
    :param quantarr: array of the image with the applied quantization
    :param n_pixels: number of pixels in image (same number in both images)
    :return: the mean squared error
    """
    return ((orgarr - quantarr) ** 2).sum() / n_pixels


def normalize(arr: np.ndarray) -> np.ndarray:
    """
    The function normalizes a pic with values between 0 and 255 to between 0 and 1
    :param arr: array to normalize
    :return: the normalized array
    """
    return arr / 255


def stretch_back(arr: np.ndarray) -> np.ndarray:
    """
    The function stretches array values from between 0 and 1 back to 0 and 255
    :param arr:
    :return:
    """
    return arr * 255


def initial_borders(histogram: np.ndarray, np_initial_seg: int, num_of_borders: int) -> np.array:
    """
    A function to get the initial borders for the quantization function
    :param histogram: the image histogram
    :param np_initial_seg: average number of pixels in each segment
    :param num_of_borders: num of borders should be in output
    :return: an array of initial borders (defining segments in image)
    """
    borders = np.zeros(num_of_borders, dtype=int)
    sum_pixels_in_each_seg = np.zeros(num_of_borders - 1, dtype=int)
    bool_dont_get_to_initial = np.zeros(num_of_borders - 1, dtype=bool)

    borders[0] = 0
    borders[num_of_borders - 1] = 255
    tmp = 0
    border_num = 1
    in_while_true = False

    for i in range(0, 256):

        # check to see if we caused a segment to be empty
        while True:
            if np.count_nonzero(histogram[borders[border_num - 1]:]) < (len(borders) - border_num - 1):
                in_while_true = True
                borders[border_num - 1] = i - 2
                bool_dont_get_to_initial[border_num - 1] = True
                sum_pixels_in_each_seg[border_num - 1] -= histogram[i - 1]
                i = i - 2

                # after fixation, the segment before is empty
                if sum_pixels_in_each_seg[border_num - 1] <= 0:  # can't be bellow zero
                    border_num -= 1
                    i = borders[border_num - 1]  # we need to fix the previous border as well
                    continue
            break

        # we did made one of the segments zero, so we want to continue to next iteration in loop
        if in_while_true:
            in_while_true = False
            continue

        # if we're on a thin line which take every histogram bin can change a lot
        # defining a border
        if bool_dont_get_to_initial[border_num - 1] and tmp > 0:
            borders[border_num] = i
            bool_dont_get_to_initial[border_num - 1] = False
            border_num += 1
            sum_pixels_in_each_seg[border_num - 2] = tmp
            tmp = 0
            continue

        # defining a border
        if tmp + histogram[i, 0] >= np_initial_seg:
            borders[border_num] = i
            bool_dont_get_to_initial[border_num - 1] = False
            border_num += 1
            sum_pixels_in_each_seg[border_num - 2] = tmp
            tmp = 0
            continue

        # didn't define a border at all at current iteration
        tmp += histogram[i]

    return borders


if __name__ == '__main__':
    img = imReadAndConvert("beach.jpg", 2)
    quantizeImage(img, 3, 9)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
