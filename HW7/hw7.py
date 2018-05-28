import cv2
import numpy as np


#
# Thinning
#


# Binarize an image
def binarize(image_data, thres):
    rows = len(image_data)
    cols = len(image_data[0])

    for i in range(rows):
        for j in range(cols):
            if (image_data[i][j] < thres):
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


# Yokoi
def yokoi(oriImg):
    rows = len(oriImg)
    cols = len(oriImg[0])

    temp = np.zeros(oriImg.shape, dtype=np.int)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            x0 = oriImg[i][j]
            if x0 == 1:
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

                temp[i][j] = f(h(x0, x1, x6, x2), h(x0, x2, x7, x3),
                               h(x0, x3, x8, x4), h(x0, x4, x5, x1))

            else:
                temp[i][j] = 7

        temp[i][0] = 7
        temp[i][cols-1] = 7

    for j in range(cols):
        temp[0][j] = 7
        temp[rows-1][j] = 7

    return temp


def pair_relation(img):
    rows = len(img)
    cols = len(img[0])
    temp = np.chararray(img.shape)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if img[i][j] != 7:
                if img[i][j] == 1 and (img[i][j + 1] == 1 or img[i - 1][j] == 1 or img[i][j - 1] == 1 or img[i + 1][j] == 1):
                    temp[i][j] = "p"
                else:
                    temp[i][j] = "q"
            else:
                temp[i][j] = "g"
        temp[i][0] = "g"
        temp[i][cols-1] = "g"

    for j in range(cols):
        temp[0][j] = "g"
        temp[rows-1][j] = "g"

    return temp


def shrink_h(b, c, d, e):
    if b != "g" and c != "g" and (d == "g" or e == "g"):
        return 1
    else:
        return 0


def shrink_f(arr, x):
    sum = 0

    for i in range(4):
        sum += arr[i]

    if sum == 1:
        return "g"
    else:
        return x


def shrink(img):
    rows = len(img)
    cols = len(img[0])
    temp = np.zeros(img.shape, dtype=np.int)
    arr = np.array([0, 0, 0, 0], dtype=np.int)

    for i in range(rows):
        for j in range(cols):
            if img[i][j] == "p":
                arr[0] = shrink_h(img[i][j], img[i][j+1],
                                  img[i-1][j+1], img[i-1][j])
                arr[1] = shrink_h(img[i][j], img[i-1][j],
                                  img[i-1][j-1], img[i][j-1])
                arr[2] = shrink_h(img[i][j], img[i][j-1],
                                  img[i+1][j-1], img[i+1][j])
                arr[3] = shrink_h(img[i][j], img[i+1][j],
                                  img[i+1][j+1], img[i][j+1])
                img[i][j] = shrink_f(arr, img[i][j])

    for i in range(rows):
        for j in range(cols):
            if img[i][j] == "g":
                temp[i][j] = 0
            else:
                temp[i][j] = 1

    return temp


def diff(img1, img2):
    rows = len(img1)
    cols = len(img1[0])

    for i in range(rows):
        for j in range(cols):
            if img1[i][j] != img2[i][j]:
                return 1

    return 0


def thinning(img):
    temp1 = np.zeros(img.shape, dtype=np.int)
    temp2 = np.chararray(img.shape)
    temp3 = np.zeros(img.shape, dtype=np.int)

    flag = True

    while(flag):
        temp1 = yokoi(img)
        temp2 = pair_relation(temp1)
        temp3 = shrink(temp2)
        flag = np.array_equal(img, temp3)
        img = temp3

    return img


if __name__ == '__main__':
    # Load an color image in grayscale
    img = cv2.imread('../image/lena.bmp', 0)
    #cv2.imshow('Input', img)

    bin_image = binarize(img, 128).astype(int)
    dsArr = np.zeros(shape=(64, 64), dtype=np.int)
    dsImg = downSampling(bin_image, dsArr, 8, 8)
    ds_rows = len(dsImg)
    ds_cols = len(dsImg[0])

    for i in range(ds_rows):
        dsImg[i][0] = 0
        dsImg[i][ds_cols-1] = 0

    for j in range(ds_cols):
        dsImg[0][j] = 0
        dsImg[ds_rows-1][j] = 0

    # Do thinning operation
    temp = thinning(dsImg)

    # Write results to text file
    text_file = open("thinning.txt", "w")

    for i in range(1, len(temp)-1):
        for j in range(1, len(temp[0])-1):
            if temp[i][j] == 0:
                text_file.write("$")
            else:
                text_file.write("*")
        text_file.write("\n")

    text_file.close()
