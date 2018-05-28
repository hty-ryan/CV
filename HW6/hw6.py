import cv2
import numpy as np

#
# Yokoi Connectivity Number
#


# Binarize an image
def binarize(image_data):
    rows = len(image_data)
    cols = len(image_data[0])

    for i in range(rows):
        for j in range(cols):
            if (image_data[i][j] < 128):
                image_data[i][j] = 0
            else:
                image_data[i][j] = 1
    return image_data


# Downsampling
def downSampling(oriImg, dsImg, unitX, unitY):
    o_rows = len(oriImg)
    o_cols = len(oriImg[0])
    d_rows = len(dsImg)
    d_cols = len(dsImg[0])

    x = 0
    for i in range(d_rows):
        y = 0
        for j in range(d_cols):
            if y < o_cols:
                dsImg[i][j] = oriImg[x][y]
            else:
                break
            y = y + unitY

        if x < o_rows:
            x = x + unitX
        else:
            break

    return dsImg


# h function
def h(b, c, d, e):
    if b == c and (d != b or e != b):
        return "q"
    elif b == c and (d == b and e == b):
        return "r"
    else:
        return "s"


# f function
def f(a1, a2, a3, a4):
    if a1 == a2 == a3 == a4 == "r":
        return 5
    else:
        n = 0
        arr = [a1, a2, a3, a4]
        for h in arr:
            if h == "q":
                n += 1

        return n


# Yokoi Connectivity Number
def yokoi(oriImg, fileName):
    text_file = open(fileName, "w")
    rows = len(oriImg)
    cols = len(oriImg[0])

    for i in range(rows):
        for j in range(cols):
            x0 = oriImg[i][j]
            if(x0 == 1):
                x1 = 0
                if j + 1 < cols:
                    x1 = oriImg[i][j+1]

                x2 = 0
                if i - 1 > 0:
                    x2 = oriImg[i-1][j]

                x3 = 0
                if j - 1 > 0:
                    x3 = oriImg[i][j-1]

                x4 = 0
                if i + 1 < rows:
                    x4 = oriImg[i+1][j]

                x5 = 0
                if i + 1 < rows and j + 1 < cols:
                    x5 = oriImg[i+1][j+1]

                x6 = 0
                if i - 1 > 0 and j + 1 < cols:
                    x6 = oriImg[i-1][j+1]

                x7 = 0
                if i - 1 > 0 and j - 1 > 0:
                    x7 = oriImg[i-1][j-1]

                x8 = 0
                if i + 1 < rows and j - 1 > 0:
                    x8 = oriImg[i+1][j-1]

                n = f(h(x0, x1, x6, x2), h(x0, x2, x7, x3),
                      h(x0, x3, x8, x4), h(x0, x4, x5, x1))

                text_file.write("%d " % n)
            else:
                text_file.write(" ")

        text_file.write("\n")

    text_file.close()

    return


# Load an color image in grayscale
img = cv2.imread('../image/lena.bmp', 0)
cv2.imshow('Input', img)

bin_image = binarize(img)
dsArr = np.zeros(shape=(64, 64))
dsImg = downSampling(bin_image, dsArr, 8, 8)

output_file = "yokoi_number.txt"

yokoi(dsImg, output_file)

cv2.imshow('DS Image', dsImg)
cv2.waitKey(0)
