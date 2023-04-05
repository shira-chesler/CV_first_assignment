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
    img = cv2.imread(filename)
    if representation == 1:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == 2:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:  # if gets wrong representation, does RGB cause it includes gray scale
        print("Got a wrong mode, converting to RGB")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    is_rgb = False
    if len(imgOrig.shape) == 3 and imgOrig.shape[2] == 3:
        # means that the image has a third dimension - RGB representation
        arr1 = transformRGB2YIQ(imgOrig)[:, :, 0]
        is_rgb = True
    elif len(imgOrig.shape) == 2:
        arr1 = imgOrig
    else:
        raise Exception("The array got is not a proper RGB/GRAY_SCALE image!")

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
    is_rgb = False
    if len(imOrig.shape) == 3 and imOrig.shape[2] == 3:
        # means that the image has a third dimension - RGB representation
        arr1 = transformRGB2YIQ(imOrig)[:, :, 0]
        is_rgb = True
    elif len(imOrig.shape) == 2:
        arr1 = imOrig
    else:
        raise Exception("The array got is not a proper RGB/GRAY_SCALE image!")

    qImage_i = []
    error_i = []
    z = np.array(nQuant+1)
    q = np.array(nQuant)

    not_norm_arr = stretch_back(arr1)  # changes values of image to be in range (0, 255)
    histogram, bin_edges = np.histogram(not_norm_arr, bins=256, range=(0, 256))


    #if nQuant>=256:
     #   return










def getWeightedMean_simple(intense: np.ndarray, cell_values: np.ndarray) -> int:
    val = np.dot(intense, cell_values)
    return val / cell_values.sum()


def normalize(arr: np.ndarray) -> np.ndarray:
    # return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr / 255


def stretch_back(arr: np.ndarray) -> np.ndarray:
    return arr * 255


if __name__ == '__main__':
    img = imReadAndConvert("beach.jpg", 2)
    hsitogramEqualize(img)
    plt.figure()
    plt.imshow(img)
    plt.show()
