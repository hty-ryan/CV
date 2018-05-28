import cv2
import numpy as np
import math


#
# Zero Crossing Edge Detection
#


# Laplacian Mask 1
def laplacian_mask1(image, threshold):
    newImg = np.zeros_like(image, dtype=np.uint8)
    temp = np.zeros(image.shape)
    mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    rows = len(image)
    cols = len(image[0])
    mask_rows = len(mask)
    mask_cols = len(mask[0])

    for i in xrange(1, rows-1):
        for j in xrange(1, cols-1):
            sum = 0
            for x in xrange(mask_rows):
                for y in xrange(mask_cols):
                    sum += image[i + x - 1][j + y - 1] * mask[x][y]
            temp[i][j] = sum

    for j in xrange(cols):
        temp[0][j] = 0
        temp[rows - 1][j] = 0

    # Zero crossing
    for i in xrange(1, rows - 1):
        for j in xrange(1, cols - 1):
            if temp[i][j] > threshold:
                for x in xrange(-1, 2):
                    for y in xrange(-1, 2):
                        if temp[i + x][j + y] < (-1 * threshold):
                            newImg[i][j] = 255

    return newImg


# Laplacian Mask 2
def laplacian_mask2(image, threshold):
    newImg = np.zeros_like(image, dtype=np.uint8)
    temp = np.zeros(image.shape)
    mask = np.array([[1.0/3, 1.0/3, 1.0/3],
                     [1.0/3, -8.0/3, 1.0/3],
                     [1.0/3, 1.0/3, 1.0/3]])
    rows = len(image)
    cols = len(image[0])
    mask_rows = len(mask)
    mask_cols = len(mask[0])

    for i in xrange(1, rows-1):
        for j in xrange(1, cols-1):
            sum = 0.0
            for x in xrange(mask_rows):
                for y in xrange(mask_cols):
                    sum += image[i+x-1][j+y-1] * mask[x][y]
            temp[i][j] = sum

    for j in xrange(cols):
        temp[0][j] = 0
        temp[rows-1][j] = 0

    # Zero crossing
    for i in xrange(1, rows-1):
        for j in xrange(1, cols-1):
            if temp[i][j] > threshold:
                for x in xrange(-1, 2):
                    for y in xrange(-1, 2):
                        if temp[i+x][j+y] < (-1 * threshold):
                            newImg[i][j] = 255

    return newImg


# Minimum variance Laplacian
def min_var_laplacian(image, threshold):
    newImg = np.zeros_like(image, dtype=np.uint8)
    temp = np.zeros(image.shape)
    mask = np.array([[2.0 / 3, -1.0 / 3, 2.0 / 3],
                     [-1.0 / 3, -3.0 / 4, -1.0 / 3],
                     [2.0 / 3, -1.0 / 3, 2.0 / 3]])
    rows = len(image)
    cols = len(image[0])
    mask_rows = len(mask)
    mask_cols = len(mask[0])

    for i in xrange(1, rows-1):
        for j in xrange(1, cols-1):
            sum = 0.0
            for x in xrange(mask_rows):
                for y in xrange(mask_cols):
                    sum += image[i+x-1][j+y-1] * mask[x][y]
            temp[i][j] = sum

    for j in xrange(cols):
        temp[0][j] = 0
        temp[rows-1][j] = 0

    # Zero crossing
    for i in xrange(1, rows - 1):
        for j in xrange(1, cols - 1):
            if temp[i][j] > threshold:
                for x in xrange(-1, 2):
                    for y in xrange(-1, 2):
                        if temp[i + x][j + y] < (-1 * threshold):
                            newImg[i][j] = 255

    return newImg


# Laplace of Gaussian
def laplace_of_gaussian(image, threshold):
    newImg = np.zeros_like(image, dtype=np.uint8)
    temp = np.zeros(image.shape)
    mask = np.array([[0, 0,  0, -1, -1, -2, -1, -1,  0,  0,  0],
                     [0, 0, -2, -4, -8, -9, -8, -4, -2,  0,  0],
                     [0, -2, -7, -15, -22, -23, -22, -15, -7, -2,  0],
                     [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
                     [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
                     [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
                     [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
                     [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
                     [0, -2, -7, -15, -22, -23, -22, -15, -7, -2,  0],
                     [0, 0, -2, -4, -8, -9, -8, -4, -2,  0,  0],
                     [0, 0,  0, -1, -1, -2, -1, -1,  0,  0,  0]
                     ])
    rows = len(image)
    cols = len(image[0])
    mask_rows = len(mask)
    mask_cols = len(mask[0])

    # Convolution with mask
    for i in xrange(5, rows-5):
        for j in xrange(5, cols-5):
            sum = 0.0
            for x in xrange(mask_rows):
                for y in xrange(mask_cols):
                    sum += image[i+x-5][j+y-5] * mask[x][y]
            temp[i][j] = sum

    for i in xrange(5):
        for j in xrange(cols):
            temp[i][j] = 0
            temp[rows-i-1][j] = 0

    # Zero crossing
    for i in xrange(5, rows - 5):
        for j in xrange(5, cols - 5):
            if temp[i][j] > threshold:
                for x in xrange(-1, 2):
                    for y in xrange(-1, 2):
                        if temp[i + x][j + y] < (-1 * threshold):
                            newImg[i][j] = 255

    return newImg


# Difference of Gaussian
def diff_of_gaussian(image, threshold):
    newImg = np.zeros_like(image, dtype=np.uint8)
    temp = np.zeros(image.shape)
    mask = np.zeros(shape=(11, 11))

    rows = len(image)
    cols = len(image[0])

    # Generate DOG mask
    sigma1 = 1.0
    sigma2 = 3.0
    mean = 0.0
    for i in xrange(-5, 6):
        for j in xrange(-5, 6):
            a = math.exp(-(i*i + j*j)/(2*sigma1*sigma1)) / \
                (math.sqrt(2*math.pi)*sigma1)
            b = math.exp(-(i*i + j*j)/(2*sigma2*sigma2)) / \
                (math.sqrt(2*math.pi)*sigma2)
            mask[i+5][j+5] = a - b
            mean += a - b

    mean /= 11*11

    for i in xrange(11):
        for j in xrange(11):
            mask[i][j] -= mean

    mask_rows = len(mask)
    mask_cols = len(mask[0])

    # Convolution with mask
    for i in xrange(5, rows-5):
        for j in xrange(5, cols-5):
            sum = 0.0
            for x in xrange(mask_rows):
                for y in xrange(mask_cols):
                    sum += image[i+x-5][j+y-5] * mask[x][y]
            temp[i][j] = sum

    for i in xrange(5):
        for j in xrange(cols):
            temp[i][j] = 0
            temp[rows-i-1][j] = 0

    # Zero crossing
    for i in xrange(5, rows - 5):
        for j in xrange(5, cols - 5):
            if temp[i][j] > threshold:
                for x in xrange(-1, 2):
                    for y in xrange(-1, 2):
                        if temp[i + x][j + y] < (-1 * threshold):
                            newImg[i][j] = 255

    return newImg


if __name__ == '__main__':
    # Load an color image in grayscale
    img = cv2.imread('../image/DOG.jpg', 0)

    lap1_img = laplacian_mask1(img, 15)
    lap2_img = laplacian_mask2(img, 15)
    min_var_lap_img = min_var_laplacian(img, 20)

    LOG_img = laplace_of_gaussian(img, 3000)
    DOG_img = diff_of_gaussian(img, 1)

    cv2.imshow('lap1_img', lap1_img)
    cv2.imshow('lap2_img', lap2_img)
    cv2.imshow('min_var_lap_img', min_var_lap_img)

    cv2.imshow('LOG_img', LOG_img)
    cv2.imshow('DOG_img', DOG_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
