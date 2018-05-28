import cv2
import numpy as np
import math
from scipy.ndimage import binary_erosion, generate_binary_structure


#
# General Edge Detection
#


EROSION_SELEM = generate_binary_structure(2, 2)


# binarize an image
def binarize(image_data, thres):
    rows = len(image_data)
    cols = len(image_data[0])

    temp_data = np.zeros_like(image_data)

    for i in range(rows):
        for j in range(cols):
            if(image_data[i][j] < thres):
                temp_data[i][j] = 255
            else:
                temp_data[i][j] = 0

    return temp_data


def convolve2d(image, kernel):
    # This function which takes an image and a kernel
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).

    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
    output = np.zeros_like(image)  # convolution output

    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image

    for x in range(image.shape[1]):  # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y:y + 3, x:x + 3]).sum()

    return output


def _mask_filter_result(result, mask):
    """Return result after masking.
    Input masks are eroded so that mask areas in the original image don't
    affect values in the result.
    """
    if mask is None:
        result[0, :] = 0
        result[-1, :] = 0
        result[:, 0] = 0
        result[:, -1] = 0
        return result
    else:
        mask = binary_erosion(mask, EROSION_SELEM, border_value=0)
        return result * mask


def roberts_operator(image):
    roberts_cross_h = np.array([[0, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]])

    roberts_cross_v = np.array([[0, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])

    vertical = convolve2d(image, roberts_cross_v)
    horizontal = convolve2d(image, roberts_cross_h)

    temp_img = np.sqrt(np.square(horizontal) + np.square(vertical))

    return temp_img


def freiChen(image, mask=None):
    return np.sqrt(freiChen_edge1(image, mask) ** 2 + freiChen_edge2(image, mask) ** 2 + freiChen_edge3(image,
                                                                                                        mask) ** 2 + freiChen_edge4(
        image, mask) ** 2 + freiChen_line1(image, mask) ** 2 + freiChen_line2(image, mask) ** 2 + freiChen_line3(image,
                                                                                                                 mask) ** 2 + freiChen_line4(
        image, mask) ** 2)


def freiChen_edge1(image, mask):

    #image = img_as_float(image)
    result = np.abs(convolve2d(image, np.array([[1, math.sqrt(2), 1],
                                                [0, 0, 0],
                                                [-1, -math.sqrt(2), -1]]).astype(float) / math.sqrt(8)))
    return _mask_filter_result(result, mask)


def freiChen_edge2(image, mask):
    #image = img_as_float(image)
    result = np.abs(convolve2d(image,
                               np.array([[1, 0, -1],
                                         [math.sqrt(2), 0, -math.sqrt(2)],
                                         [1, 0, -1]]).astype(float) / math.sqrt(8)))
    return _mask_filter_result(result, mask)


def freiChen_edge3(image, mask):
    #image = img_as_float(image)
    result = np.abs(convolve2d(image, np.array(
        [[0, -1, math.sqrt(2)], [1, 0, -1], [-math.sqrt(2), 1, 0]]).astype(float) / math.sqrt(8)))
    return _mask_filter_result(result, mask)


def freiChen_edge4(image, mask):
    # image = img_as_float(image)
    result = np.abs(convolve2d(image, np.array([[math.sqrt(
        2), -1, 0], [-1, 0, 1], [0, 1, -math.sqrt(2)]]).astype(float) / math.sqrt(8)))
    return _mask_filter_result(result, mask)


def freiChen_line1(image, mask):
    #image = img_as_float(image)
    result = np.abs(convolve2d(image, np.array(
        [[0, 1, 0], [-1, 0, -1], [0, 1, 0]]).astype(float) / 2.0))
    return _mask_filter_result(result, mask)


def freiChen_line2(image, mask):
    # image = img_as_float(image)
    result = np.abs(convolve2d(image, np.array(
        [[-1, 0,  1], [0, 0,  0], [1, 0, -1]]).astype(float) / 2.0))
    return _mask_filter_result(result, mask)


def freiChen_line3(image, mask):
    #image = img_as_float(image)
    result = np.abs(convolve2d(image, np.array(
        [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]).astype(float) / 6.0))
    return _mask_filter_result(result, mask)


def freiChen_line4(image, mask):
    #image = img_as_float(image)
    result = np.abs(convolve2d(image, np.array(
        [[-2, 1, -2], [1, 4, 1], [-2, 1, -2]]).astype(float) / 6.0))
    return _mask_filter_result(result, mask)


def freiChen_average(image, mask):
    #image = img_as_float(image)
    result = np.abs(convolve2d(image, np.array(
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(float) / 3.0))
    return _mask_filter_result(result, mask)


if __name__ == '__main__':
    # Load an color image in grayscale
    img = cv2.imread('../image/lena.bmp', 0)

    #kirsch_img = binarize(img, 135)
    #robinson_img = binarize(img, 43)
    #babu_img = binarize(img, 100)

    #cv2.imshow('roberts_img', roberts_operator_img)
    #cv2.imshow('prewitt_operator_img', prewitt_operator_img)
    #cv2.imshow('sobel_operator_img', sobel_operator_img)
    #cv2.imshow('frei_chen_img', frei_chen_img)
    #cv2.imshow('kirsch_img', kirsch_img)
    #cv2.imshow('robinson_img', robinson_img)
    #cv2.imshow('babu_img', babu_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
