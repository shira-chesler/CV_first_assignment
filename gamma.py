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
from ex1_utils import LOAD_GRAY_SCALE
from __future__ import division
from __future__ import print_function

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    pass


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()

import cv2 as cv
import argparse

alpha_slider_max = 100
title_window = 'Linear Blend'


def on_trackbar(val):
    alpha = val / alpha_slider_max
    beta = (1.0 - alpha)
    dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    cv.imshow(title_window, dst)


parser = argparse.ArgumentParser(description='Code for Adding a Trackbar to our applications tutorial.')
parser.add_argument('--input1', help='Path to the first input image.', default='LinuxLogo.jpg')
parser.add_argument('--input2', help='Path to the second input image.', default='WindowsLogo.jpg')
args = parser.parse_args()
src1 = cv.imread("beach.jpg")
src2 = cv.imread("beach.jpg")
if src1 is None:
    print('Could not open or find the image: ', args.input1)
    exit(0)
if src2 is None:
    print('Could not open or find the image: ', args.input2)
    exit(0)
cv.namedWindow(title_window)
trackbar_name = 'Alpha x %d' % alpha_slider_max
cv.createTrackbar(trackbar_name, title_window, 0, 2, on_trackbar)
# Show some stuff
on_trackbar(0)
# Wait until user press some key
cv.waitKey()
