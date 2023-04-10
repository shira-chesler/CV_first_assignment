# First Assignment in Computer Vision course

I used Python version 3.9 and PyCharm platform to test my
program. 

### The files I'm submitting are:
* ex1_main.py - The main given by lecturer in order to test the program.
* gamma.py - The file with the gamma correction GUI slider
* ex1_utils - The file with the rest of the functions
* bac_con.png, beach.jpg, dark.jpg, water_bear.png, testImg1.jpg, testImg2.jpg - images that can be used to test the code on.

### The functions I've written with a short description of each:
inside ex1_utils:
* myID - Return my ID.
* imReadAndConvert - Reads an image, and returns the image converted as requested.
* imDisplay - Reads an image as RGB or GRAY_SCALE and displays it.
* transformRGB2YIQ - Converts an RGB image to YIQ color space.
* transformYIQ2RGB - Converts an YIQ image to RGB color space.
* hsitogramEqualize - Equalizes the histogram of an image.
* quantizeImage - Quantized an image in to **nQuant** (parameter given) colors.
* image_type_check - The function checks is the image is RGB or gray scale based on the shape of the image.
* getWeightedMean - The function calculates the weighted mean of each one of the given segments (given by segments borders).
* getNewBorders - The function calculates the new borders by the mean of the two weighted mean values given.
* apply_quant - The function applies a given quantization.
* calc_mse - The function calculates the mse between the original image arr to the applied quantization one.
* normalize - The function normalizes a pic with values between 0 and 255 to between 0 and 1.
* stretch_back - The function stretches array values from between 0 and 1 back to 0 and 255.
* initial_borders - A function to get the initial borders for the quantization function.

inside gamma.py:
* nothing - A function that does nothing (necessary for track bar).
* apply_power - The function applies a wanted power to an numpy ndarray representation of an image.
* imReadAndConvert - Reads an image, and returns the image converted as requested (but RGB as BGR).
* gammaDisplay - GUI for gamma correction.
