import numpy as np
import cv2


# Load an color image
img = cv2.imread('../image/lena.bmp', 0)

image_data = np.asarray(img, dtype=np.uint8)


# Flipping an image horizontally
def flip_horizontal(image_data):
    rows = len(image_data)
    cols = len(image_data[0])

    for i in range(rows):
        for j in range(cols/2):
            temp = image_data[i][j]
            image_data[i][j] = image_data[i][cols-1-j]
            image_data[i][cols - 1 - j] = temp
    return image_data

# Flipping an image vertically


def flip_vertical(image_data):
    rows = len(image_data)
    cols = len(image_data[0])

    for i in range(rows / 2):
        for j in range(cols):
            temp = image_data[i][j]
            image_data[i][j] = image_data[rows-1-i][j]
            image_data[rows - 1 - i][j] = temp
    return image_data

# Flipping an image diagonally


def flip_diagonal(image_data):
    rows = len(image_data)
    cols = len(image_data[0])

    for i in range(rows):
        for j in range(i):
            temp = image_data[i][j]
            image_data[i][j] = image_data[rows-i][cols - 1 - j]
            image_data[rows-i][cols - 1 - j] = temp
    return image_data


#image_data = flip_horizontal(image_data)
#image_data = flip_vertical(image_data)
image_data = flip_diagonal(image_data)

cv2.imwrite('color_img.jpg', image_data)
cv2.imshow('Color image', image_data)

cv2.waitKey(0)
cv2.destroyAllWindows()
