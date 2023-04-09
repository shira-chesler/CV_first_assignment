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
    lut = np.ceil((cum_sum * 255) / (arr1.shape[0] * arr1.shape[1])).astype(int)  # lookup table
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
    # z = np.zeros(nQuant + 1)
    # # q = np.array(nQuant)

    not_norm_arr = stretch_back(arr1)  # changes values of image to be in range (0, 255)
    histogram, bin_edges = np.histogram(not_norm_arr, bins=256, range=(0, 256))
    histogram = histogram.reshape(256, 1)
    bin_edges = bin_edges[:256].reshape(256, 1)
    # print(histogram.shape)
    # print(bin_edges.shape)

    if np.count_nonzero(histogram) <= nQuant:
        return imOrig

    num_of_pixels = imOrig.shape[0] * imOrig.shape[1]
    np_initial_seg = num_of_pixels / nQuant
    z = initial_borders(histogram, np_initial_seg, nQuant + 1)

    for i in range(0, nIter):
        q = getWeightedMean(bin_edges, histogram, z)
        z = getNewBorders(q)
        # print("means",q)
        # print("borders", z)
        new_image = apply_quant(q, z, np.copy(not_norm_arr))
        # sec_new_img = gpt_apply_quant(q, z, not_norm_arr.copy())
        # print(np.array_equal(sec_new_img, new_image))
        mse = calc_mse(not_norm_arr, new_image, num_of_pixels)
        error_i.append(mse)
        if is_rgb:
            tmp = transformRGB2YIQ(stretch_back(imOrig))
            tmp[:, :, 0] = new_image
            new_image = transformYIQ2RGB(tmp)
        qImage_i.append(normalize(new_image))
        # print(error_i)
        if mse == 0 or (len(error_i) > 2 and mse in error_i):  # converges
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
    means = np.empty(len(borders) - 1)
    borders = np.ceil(borders).astype(int)
    for i in range(0, len(borders) - 1):
        val = np.dot(intensity[borders[i]:borders[i + 1]].T, pixels_in_intensity[borders[i]:borders[i + 1]])
        n_pixels = pixels_in_intensity[borders[i]:borders[i + 1]].sum()
        means[i] = val / n_pixels
        # val = 0
        # for j in range(len(intense[borders[i]:borders[i + 1]])):
        #     val += (intense[borders[i]:borders[i + 1]])[j] * (pixels_in_intense[borders[i]:borders[i + 1]])[j]
        # print(val / ((pixels_in_intense[borders[i]:borders[i + 1]]).sum()) == means[i])
        # print("val:", val / pixels_in_intense[borders[i]:borders[i + 1]].sum(), "mean: ", means[i])

    return means


def getNewBorders(seg_values: np.ndarray) -> np.ndarray:
    shift = np.roll(seg_values, 1)
    # print("shift:", shift)
    # print("orig:", seg_values)
    new_borders = (shift + seg_values) / 2
    new_borders[0] = 0
    new_borders = np.append(new_borders, 255)
    # print("new borders:", new_borders)
    return new_borders


def apply_quant(values: np.ndarray, borders: np.ndarray, org_im: np.ndarray) -> np.ndarray:
    for i in range(0, len(values)):
        # print(org_im.shape)
        org_im[(org_im >= borders[i]) & (org_im <= borders[i + 1])] = values[i]
        # org_im = np.where((org_im >= borders[i]) and (org_im <= borders[i + 1]), values[i], org_im)
    return org_im


def gpt_apply_quant(values: np.ndarray, borders: np.ndarray, org_im: np.ndarray) -> np.ndarray:
    org_im[np.logical_and(org_im >= borders[:-1], org_im <= borders[1:])] = values[np.arange(len(values))]
    return org_im


def calc_mse(orgarr: np.ndarray, quantarr: np.ndarray, n_pixels: int) -> int:
    return ((orgarr - quantarr) ** 2).sum() / n_pixels


def normalize(arr: np.ndarray) -> np.ndarray:
    # return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr / 255


def stretch_back(arr: np.ndarray) -> np.ndarray:
    return arr * 255


def initial_borders(histogram: np.ndarray, np_initial_seg: int, num_of_borders: int) -> np.array:
    borders = np.zeros(num_of_borders, dtype=int)
    sum_pixels_in_each_seg = np.zeros(num_of_borders - 1, dtype=int)
    bool_dont_get_to_initial = np.zeros(num_of_borders - 1, dtype=bool)
    borders[0] = 0
    borders[num_of_borders - 1] = 255
    tmp = 0
    border_num = 1
    in_while_true = False
    for i in range(0, 256):
        while True:
            if np.count_nonzero(histogram[borders[border_num - 1]:]) < (len(borders) - border_num - 1):
                in_while_true = True
                borders[border_num - 1] = i - 2
                bool_dont_get_to_initial[border_num - 1] = True
                sum_pixels_in_each_seg[border_num - 1] -= histogram[i - 1]
                i = i - 2
                if sum_pixels_in_each_seg[border_num - 1] <= 0:  # can't be bellow zero
                    border_num -= 1
                    i = borders[border_num - 1]
                    continue
            break
        if in_while_true:
            in_while_true = False
            continue
        if bool_dont_get_to_initial[border_num - 1] and tmp > 0:
            borders[border_num] = i
            bool_dont_get_to_initial[border_num - 1] = False
            border_num += 1
            sum_pixels_in_each_seg[border_num - 2] = tmp
            tmp = 0
            continue
        if tmp + histogram[i, 0] >= np_initial_seg:
            borders[border_num] = i
            bool_dont_get_to_initial[border_num - 1] = False
            border_num += 1
            sum_pixels_in_each_seg[border_num - 2] = tmp
            tmp = 0
            continue
        tmp += histogram[i]
    return borders


if __name__ == '__main__':
    img = imReadAndConvert("beach.jpg", 2)
    quantizeImage(img, 3, 9)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
