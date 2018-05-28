import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load an color image in grayscale
img = cv2.imread('../image/lena.bmp', 0)
image_data = np.asarray(img, dtype=np.uint8)
bins = np.zeros(256)

# plt.hist(img.ravel(),256,[0,256])
# plt.show()


# binarize an image
def binarize(image_data):
    rows = len(image_data)
    cols = len(image_data[0])

    for i in range(rows):
        for j in range(cols):
            if(image_data[i][j] < 128):
                image_data[i][j] = 0
            else:
                image_data[i][j] = 255
    return image_data


# Calculate and draw a histogram
def histogram(image_data, bins):
    rows = len(image_data)
    cols = len(image_data[0])

    for i in range(rows):
        for j in range(cols):
            pix = image_data[i][j]
            bins[pix] += 1

    plt.bar(range(len(bins)), bins, color="blue")

    plt.show()

    return


histogram(image_data, bins)

#image_data = binarize(image_data)

#cv2.imwrite('color_img.jpg', image_data)
#cv2.imshow('Color image', image_data)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
