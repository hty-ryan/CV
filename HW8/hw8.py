import cv2
import numpy as np


#
# Noise Removal (4 resource pictures + 24 result pictures)
#


# Add Gaussian noise
def gaussian_noise(image, amp):
    temp_image = np.float64(np.copy(image))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * amp

    noisy_image = np.zeros(temp_image.shape, np.float64)
    noisy_image = temp_image + noise

    #print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    #print('type = ', type(noisy_image[0][0][0]))
    return noisy_image


# Add salt and pepper noise
def salt_and_pepper_noise(image, thres):
    output = np.zeros(image.shape, np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.uniform(0.0, 1.0, None)
            if rdn < thres:
                output[i][j] = 0
            elif rdn > 1 - thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def convert_to_uint8(image_in):
    temp_image = np.float64(np.copy(image_in))
    cv2.normalize(temp_image, temp_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)

    return temp_image.astype(np.uint8)


def bubbleSort(alist):
    for passnum in range(len(alist) - 1, 0, -1):
        for i in range(passnum):
            if alist[i] > alist[i + 1]:
                temp = alist[i]
                alist[i] = alist[i + 1]
                alist[i + 1] = temp


# Box filter
def box_filter(img, kernel, ox, oy):
    rows = len(img)
    cols = len(img[0])
    se_row = kernel.shape[0]
    se_col = kernel.shape[1]
    temp = np.zeros_like(img)

    for i in range(rows):
        for j in range(cols):
            sum = 0
            count = 0
            for m in range(se_row):
                for n in range(se_col):
                    if (i - ox + m >= 0 and j - oy + n >= 0 and i - ox + m < rows and j - oy + n < cols):
                        if kernel[m][n] == 1:
                            sum += img[i - ox + m][j - oy + n]
                            count += 1
            temp[i][j] = sum/count

    return temp


# Median filter
def median_filter(img, kernel, ox, oy):
    rows = len(img)
    cols = len(img[0])
    se_row = kernel.shape[0]
    se_col = kernel.shape[1]
    temp_img = np.zeros_like(img)
    arr = np.zeros(se_row * se_col)

    for i in range(rows):
        for j in range(cols):
            k = 0
            for m in range(se_row):
                for n in range(se_col):
                    if (i - ox + m < rows and j - oy + n < cols):
                        arr[k] = img[abs(i - ox + m)][abs(j - oy + n)]
                    elif (i - ox + m >= rows and j - oy + n < cols):
                        arr[k] = img[2 * (rows - 1) - (i - ox + m)][j - oy + n]
                    elif (i - ox + m < rows and j - oy + n >= cols):
                        arr[k] = img[i - ox + m][2 * (cols - 1) - (j - oy + n)]
                    else:
                        arr[k] = 0
                    k = k + 1
            arr.sort()
            temp_img[i][j] = arr[(k - 1) / 2]

    return temp_img


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


def opening_then_closing_filter(img, kernel, ox, oy):
    temp = opening(img, kernel, ox, oy)
    temp = closing(temp, kernel, ox, oy)

    return temp


def closing_then_opening_filter(img, kernel, ox, oy):
    temp = closing(img, kernel, ox, oy)
    temp = opening(temp, kernel, ox, oy)

    return temp


if __name__ == '__main__':
    # Load an color image in grayscale
    img = cv2.imread('../image/lena.bmp', 0)

    # Create noise
    #noise_img_10 = convert_to_uint8(gaussian_noise(img, 10))
    #noise_img_30 = convert_to_uint8(gaussian_noise(img, 30))
    noise_img_005 = salt_and_pepper_noise(img, 0.05)
    noise_img_01 = salt_and_pepper_noise(img, 0.1)

    #cv2.imshow('gaussian_noise_10', noise_img_10)
    #cv2.imshow('gaussian_noise_30', noise_img_30)
    cv2.imshow('salt_and_pepper_005', noise_img_005)
    cv2.imshow('salt_and_pepper_01', noise_img_01)

    # Create box filter kernel
    #kernel = np.ones((3,3), dtype=np.int)
    #kernel = np.ones((5,5), dtype=np.int)
    kernel = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [
                      1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]])

    # Original of kernel
    ox = (kernel.shape[0] - 1) / 2
    oy = (kernel.shape[1] - 1) / 2

    # Gaussian
    #box_filter_image_gaussian_10 = box_filter(noise_img_10, kernel, 2, 2)
    #box_filter_image_gaussian_30 = box_filter(noise_img_30, kernel, 2, 2)
    #median_filter_image_gaussian_10 = median_filter(noise_img_10, kernel, ox, oy)
    #median_filter_image_gaussian_30 = median_filter(noise_img_30, kernel, ox, oy)
    #opening_then_closing_gaussian_10 = opening_then_closing_filter(noise_img_10, kernel, 2, 2)
    #opening_then_closing_gaussian_30 = opening_then_closing_filter(noise_img_30, kernel, 2, 2)
    #closing_then_opening_gaussian_10 = closing_then_opening_filter(noise_img_10, kernel, 2, 2)
    #closing_then_opening_gaussian_30 = closing_then_opening_filter(noise_img_30, kernel, 2, 2)

    # Salt and pepper
    #box_filter_image_sp_005 = box_filter(noise_img_005, kernel, 2, 2)
    #box_filter_image_sp_01 = box_filter(noise_img_01, kernel, 2, 2)
    #median_filter_image_sp_005 = median_filter(noise_img_005, kernel, ox, oy)
    #median_filter_image_sp_01 = median_filter(noise_img_01, kernel, ox, oy)
    #opening_then_closing_sp_005 = opening_then_closing_filter(noise_img_005, kernel, 2, 2)
    #opening_then_closing_sp_01 = opening_then_closing_filter(noise_img_01, kernel, 2, 2)
    closing_then_opening_sp_005 = closing_then_opening_filter(
        noise_img_005, kernel, 2, 2)
    closing_then_opening_sp_01 = closing_then_opening_filter(
        noise_img_01, kernel, 2, 2)

    #cv2.imshow('box_filter_image_gussian_10', box_filter_image_gaussian_10)
    #cv2.imshow('box_filter_image_gussian_30', box_filter_image_gaussian_30)
    #cv2.imshow('box_filter_image_sp_005', box_filter_image_sp_005)
    #cv2.imshow('box_filter_image_sp_01', box_filter_image_sp_01)
    #cv2.imshow('median_filter_image_gaussian_10', median_filter_image_gaussian_10)
    #cv2.imshow('median_filter_image_gaussian_30', median_filter_image_gaussian_30)
    #cv2.imshow('median_filter_image_sp_005', median_filter_image_sp_005)
    #cv2.imshow('median_filter_image_sp_01', median_filter_image_sp_01)
    #cv2.imshow('opening_then_closing_gaussian_10', opening_then_closing_gaussian_10)
    #cv2.imshow('opening_then_closing_gaussian_30', opening_then_closing_gaussian_30)
    #cv2.imshow('opening_then_closing_sp_005', opening_then_closing_sp_005)
    #cv2.imshow('opening_then_closing_sp_01', opening_then_closing_sp_01)
    #cv2.imshow('closing_then_opening_gaussian_10', closing_then_opening_gaussian_10)
    #cv2.imshow('closing_then_opening_gaussian_30', closing_then_opening_gaussian_30)
    cv2.imshow('closing_then_opening_sp_005', closing_then_opening_sp_005)
    cv2.imshow('closing_then_opening_sp_01', closing_then_opening_sp_01)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
