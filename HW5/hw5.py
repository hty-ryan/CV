import cv2
import numpy as np

#
# Mathematical Morphology - Gray Scaled Morphology
#


def bubbleSort(alist):
    for passnum in range(len(alist) - 1, 0, -1):
        for i in range(passnum):
            if alist[i] > alist[i + 1]:
                temp = alist[i]
                alist[i] = alist[i + 1]
                alist[i + 1] = temp


# Dilation
def dilation(img, kernel, ox, oy):
    rows = len(img)
    cols = len(img[0])
    se_row = kernel.shape[0]
    se_col = kernel.shape[1]

    temp = np.zeros_like(img)
    arr = np.zeros(se_row * se_col)

    for i in range(rows):
        for j in range(cols):
            k = 0
            for m in range(se_row):
                for n in range(se_col):
                    if (i - ox + m >= 0 and j - oy + n >= 0 and i - ox + m < rows and j - oy + n < cols):
                        if kernel[m][n] == 1:
                            arr[k] = img[i - ox + m][j - oy + n]
                            k = k + 1
            bubbleSort(arr)
            temp[i][j] = arr[k - 1]

    return temp


# Erosion
def erosion(img, kernel, ox, oy):
    rows = len(img)
    cols = len(img[0])
    se_row = kernel.shape[0]
    se_col = kernel.shape[1]

    temp = np.zeros_like(img)
    arr = np.zeros(se_row * se_col)

    for i in range(rows):
        for j in range(cols):
            k = 0
            for m in range(se_row):
                for n in range(se_col):
                    if (i - ox + m >= 0 and j - oy + n >= 0 and i - ox + m < rows and j - oy + n < cols):
                        if kernel[m][n] == 1:
                            arr[k] = img[i - ox + m][j - oy + n]
                            k = k + 1
            bubbleSort(arr)
            temp[i][j] = arr[0]

    return temp

# Opening


def opening(img, kernel, ox, oy):
    temp = np.zeros_like(img)
    temp = erosion(img, kernel, ox, oy)
    temp = dilation(temp, kernel, ox, oy)

    return temp


# Closing
def closing(img, kernel, ox, oy):
    temp = np.zeros_like(img)
    temp = dilation(img, kernel, ox, oy)
    temp = erosion(temp, kernel, ox, oy)

    return temp


# Load an color image in grayscale
img = cv2.imread('../image/lena.bmp', 0)
cv2.imshow('Input', img)

kernel = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [
                  1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]])

#img_dilation = dilation(img, kernel, 2, 2)
#img_erotion = erosion(img, kernel, 2, 2)
#img_opening = opening(img, kernel, 2, 2)
img_closing = closing(img, kernel, 2, 2)

#cv2.imshow('Dilation', img_dilation)
#cv2.imshow('Erotion', img_erotion)
#cv2.imshow('Opening', img_opening)
cv2.imshow('Closing', img_closing)
cv2.waitKey(0)
