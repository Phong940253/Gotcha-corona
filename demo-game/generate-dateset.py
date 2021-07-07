import os
import numpy as np
from skimage import color
import skimage.io as io
import skimage.transform as transform
from skimage.color import rgba2rgb
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import pandas as pd


def imgAug(img):
    sizeRandom = random.uniform(0.2, 0.8)
    res = cv.resize(img, None, fx=sizeRandom, fy=sizeRandom)
    return res


def inL(x1, x2, x3, x4):
    # check x1 <= x2, x3 <= x4  x2 <= x1, x4 <= x3
    return (x1 <= x2 and x2 <= x4) or (x1 <= x3 and x3 <= x4) or (x2 <= x1 and x1 <= x3) or (x2 <= x4 and x4 <= x3)


def compose(corona, background, info):
    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = corona.shape
    rowsb, colsb, channelsb = background.shape

    while True:
        passWhile = True
        min_y, min_x = random.randint(
            0, rowsb - rows), random.randint(0, colsb - cols)
        max_y, max_x = rows + min_y, cols + min_x

        for i in info:
            rect, type = i
            x1, x2, y1, y2 = rect
            if inL(x1, min_x, max_x, x2) and inL(y1, min_y, max_y, y2):
                x = [x1, x2, min_x, max_x]
                y = [y1, y2, min_y, max_y]
                x.sort()
                y.sort()
                if (x[2] - x[1]) * (y[2] - y[1]) >= 0.75 * (x2 - x1) * (y2 - y1):
                    passWhile = False
                    print("Re-generate")
                    break
        if passWhile:
            break
    # check new corona in old corona

    roi = background[min_y: max_y, min_x: max_x]

    # print(min_x, max_x, min_y, max_y, background.shape)
    # print(roi.shape, rows, cols)
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv.cvtColor(corona, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 20, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv.bitwise_and(corona, corona, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv.add(img1_bg, img2_fg)
    background[min_y: max_y, min_x: max_x] = dst

    return background, (min_x, max_x, min_y, max_y)


def getForegroundMask(foreground):
    mask_new = foreground.copy()[:, :, 1]
    mask_new[mask_new > 0 and mask_new < 1] = 1
    return mask_new


def drawBoundingBox(img, info):
    for i in info:
        rect, type = i
        min_x, max_x, min_y, max_y = rect
        # print(min_x, max_x, min_y, max_y)
        type = type[:-4]
        if type == "doctor" or type == "patient":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv.rectangle(img, (min_x, min_y), (max_x, max_y),  color, 2)
        cv.putText(img, type, (min_x, min_y - 5), 0, 0.3, color)
    return img


mapLabel = {
    "type1": 0,
    "type2": 0,
    "type3": 0,
    "queen": 0,
    "patient": 1,
    "doctor": 1
}

data = ["image_id", "width", "height", "bbox", "labels", "bg"]


def addLabelToFile(df, info, filename, size, bg_name):
    for i in info:
        bbox = list(i[0])
        d = {"image_id": filename,
             "width": size[0], "height": size[1], "bbox": [bbox], "labels": mapLabel[i[1][:-4]], "bg": [bg_name]}
        df2 = pd.DataFrame(data=d)
        df = df.append(df2)
    return df


dict = os.listdir("./Corona/background/")
df = pd.DataFrame(columns=data)

for ind in range(0, 1000):
    background_filename = np.random.choice(os.listdir("./Corona/background/"))
    background = cv.imread('./Corona/background/' + background_filename)
    # filename = './Corona/background/' + val
    print(background_filename)
    info = []
    mu = 8
    sigma = 4
    for i in range(0, int(random.gauss(mu, sigma))):
        corona_filename = np.random.choice(os.listdir("./Corona/corona/"))
        corona = cv.imread('./Corona/corona/' + corona_filename)
        corona_aug = imgAug(corona)
        background, rect = compose(
            corona=corona_aug, background=background, info=info)
        info.append((rect, corona_filename))
        # cv.imshow('res', background)
        # cv.waitKey()

    cv.imwrite('./Corona/train/' + str(ind) + '.jpg', background)
    background = drawBoundingBox(background, info)
    cv.imwrite('./Corona/labled/' + str(ind) + '.jpg', background)
    df = addLabelToFile(
        df, info, ind, background.shape[:2], background_filename)
    # cv.imshow('res', background)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # mask_new = getForegroundMask(corona_rgb)
    # corona_rgb[corona_rgb >= 1] = 0
    # composed_image = compose(foregroundAug(corona_rgb), mask_new, background)
    # plt.axis('off')
    # plt.imshow(mask_new)
    # plt.show()
print(df.head())
df.to_csv('./Corona/train.csv', index=False)
# nums = []
# for i in range(1000):
#     temp = random.gauss(15, 5)
#     nums.append(temp)

# # plotting a graph
# plt.hist(nums, bins=200)
# plt.show()

# plt.axis('off')
# plt.imshow(corona)
# plt.show()
