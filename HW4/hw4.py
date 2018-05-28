import cv2
import numpy as np

#
# Mathematical Morphology - Binary Morphology
#


#binarize an image
def binarize(image_data):
    rows = len(image_data)
    cols = len(image_data[0])

    for i in range(rows):
        for j in range(cols):
            if(image_data[i][j] < 128):
                image_data[i][j] = 0
            else:
                image_data[i][j] = 1
    return image_data


def grey(image_data):
    rows = len(image_data)
    cols = len(image_data[0])

    for i in range(rows):
        for j in range(cols):
            if(image_data[i][j] == 1):
                image_data[i][j] = 255

    return image_data


def compliment(image_data):
    rows = len(image_data)
    cols = len(image_data[0])

    for i in range(rows):
        for j in range(cols):
            if(image_data[i][j] == 1):
                image_data[i][j] = 0
            else:
                image_data[i][j] = 1

    return image_data


#Dilation
def dilation(img, kernel, ox, oy):
    rows = len(img)
    cols = len(img[0])
    se_row = kernel.shape[0]
    se_col = kernel.shape[1]

    temp = np.zeros_like(img)

    for i in range(rows):
        for j in range(cols):
            if(img[i][j] == 1):
                #covered by se
                for m in range(se_row):
                    for n in range(se_col):
                        if(i-ox+m >= 0 and j-oy+n >= 0 and i-ox+m < rows and j-oy+n < cols):
                            if(temp[i-ox+m][j-oy+n] == 0):
                                temp[i - ox + m][j - oy + n] = kernel[m][n]

    return temp


#Erosion
def erosion(img, kernel, ox, oy):
    rows = len(img)
    cols = len(img[0])
    se_row = kernel.shape[0]
    se_col = kernel.shape[1]

    temp = np.zeros_like(img)

    sum = 0
    for m in range(se_row):
        for n in range(se_col):
            if(kernel[m][n] == 1):
                sum+=1

    for i in range(rows):
        for j in range(cols):
            if(img[i][j] == 1):
                count = 0
                for m in range(se_row):
                    for n in range(se_col):
                        if (i - ox + m >= 0 and j - oy + n >= 0 and i - ox + m < rows and j - oy + n < cols):
                            count += (img[i - ox + m][j - oy + n] & kernel[m][n])

                if(sum == count):
                    temp[i][j] = 1

    return temp


#Opening
def opening(img, kernel, ox, oy):
    temp = np.zeros_like(img)
    temp = erosion(img, kernel, ox, oy)
    temp = dilation(temp, kernel, ox, oy)

    return temp


#Closing
def closing(img, kernel, ox, oy):
    temp = np.zeros_like(img)
    temp = dilation(img, kernel, ox, oy)
    temp = erosion(temp, kernel, ox, oy)

    return temp


def hit_and_miss(img, kernel_j, kernel_k,  k_ox, k_oy, j_ox, j_oy):
    temp1 = np.zeros_like(img)
    temp2 = np.zeros_like(img)
    temp3 = np.zeros_like(img)

    temp1 = erosion(img, kernel_j, j_ox, j_oy)
    temp2 = erosion(compliment(img), kernel_k, k_ox, k_oy)

    for i in range(len(temp3)):
        for j in range(len(temp3[0])):
            if(temp1[i][j] == 1 and temp2[i][j] == 1):
                temp3[i][j] = 1

    return temp3


kernel = np.array([[0,1,1,1,0], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [0,1,1,1,0]])
kernel_j = np.array([[0,0,0,0,0], [0,0,0,0,0], [1,1,0,0,0], [0,1,0,0,0], [0,0,0,0,0]])
kernel_k = np.array([[0,0,0,0,0], [0,1,1,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0]])
#kernel = np.ones((5,5), np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))


# Load an color image in grayscale
img = cv2.imread('../image/lena.bmp',0)
cv2.imshow('Input', img)

img_bin = binarize(img)

#img_dilation = dilation(img, kernel, 2, 2)
#img_erosion = erosion(img, kernel, 2, 2)
#img_erosionk = erosion(compliment(img), kernel_k, 2, 1)
#img_opening = opening(img_bin, kernel, 2, 2)
img_closing = closing(img, kernel, 2, 2)
#img_hm = hit_and_miss(img, kernel_j, kernel_k, 2, 1, 2, 1)

#img_dilation = grey(img_dilation)
#img_erosion = grey(img_erosion)
#img_erosionk = grey(img_erosionk)
#img_opening = grey(img_opening)
img_closing = grey(img_closing)
#img_hm = grey(img_hm)

#cv2.imshow('Dilation', img_dilation)
#cv2.imshow('Erosion', img_erosion)
#cv2.imshow('Erosionk', img_erosionk)
#cv2.imshow('Opening', img_opening)
cv2.imshow('Closing', img_closing)
#cv2.imshow('Hit_and_Miss', img_hm)

cv2.waitKey(0)

