import numpy as np
import cv2
from matplotlib import pyplot as plt

#
# Histogram
#

# Calculates histogram of an image


def histogram(im):
    x, y = im.shape
    bins = np.zeros(256)

    for i in range(x):
        for j in range(y):
            bins[im[i, j]] += 1

    return np.array(bins) / (x * y)


# Finds cumulative sum of a numpy array, list
def cum_sum(h):
    return [sum(h[:i+1]) for i in range(len(h))]


# Calculate histogram equalization
def histeq(im):
    # calculate normalized Histogram
    #m, n = im.shape
    h = histogram(im)

    # cumulative distribution function
    cdf = np.array(cum_sum(h))

    # Transform function
    tf = np.uint8(255 * cdf)
    m, n = im.shape
    Y = np.zeros_like(im)

    rows = len(Y)
    cols = len(Y[0])

    for i in range(rows):
        for j in range(cols):
            Y[i, j] = tf[im[i, j]]

    H = histogram(Y)

    return Y, h, H


# Load an color image in grayscale
img = cv2.imread('../image/lena.bmp', 0)
image_data = np.asarray(img, dtype=np.uint8)

# Get histogram equalization
new_image, o_histogram, eq_histogram = histeq(image_data)

# Draw original histogram
plt.bar(range(len(o_histogram)), o_histogram, color="blue")
plt.show()

# Draw histogram equalization
plt.bar(range(len(eq_histogram)), eq_histogram, color="blue")
# plt.plot(histeq(image_data))
plt.show()

# Draw new image
cv2.imwrite('color_img.jpg', new_image)
cv2.imshow('Color image', new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
