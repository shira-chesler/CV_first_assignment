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


import numpy as np

from ex1_utils import LOAD_GRAY_SCALE, normalize

import cv2 as cv2

slider_max = 200
title_window = 'Gamma Correction'


def nothing(x):
    """
    A function that does nothing (necessary for track bar)
    :param x: the location that the track bar sends to the function
    """
    pass


def apply_power(image: np.ndarray, power: float) -> np.ndarray:
    """
    The function applies a wanted power to an numpy ndarray representation of an image
    :param image: numpy nd array representation of the image we want to apply the power to
    :param power: the gamma power we need to apply to the image
    :return: an np ndarray representation of the image with the power applied to it
    """
    return np.power(image, power)


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested (but RGB as BGR)
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    image = cv2.imread(filename)
    while True:
        if representation == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif representation == 2:
            image = image
        else:
            representation = int(input("Got wrong mode, please renter - 1 or 2"))
            continue
        break
    arr_im = np.array(image).astype(np.float64)
    arr_norm = normalize(arr_im)
    return arr_norm


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    src = imReadAndConvert(img_path, rep)
    if src is None:
        print('Could not open or find the image: ', str)
        exit(0)
    cv2.namedWindow(title_window)
    trackbar_name = 'Gamma'
    new = src
    cv2.imshow(title_window, new)
    cv2.createTrackbar(trackbar_name, title_window, 0, slider_max, nothing)

    while True:
        # show image
        cv2.imshow(title_window, new)

        # for button pressing and changing
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of the trackbar
        val = cv2.getTrackbarPos(trackbar_name, title_window)

        # apply gamma power according to track bar position
        new = apply_power(src, val * 0.01)

    cv2.waitKey()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
