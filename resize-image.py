import os
import cv2 as cv
import random

dict = os.listdir("./gotcha-corona-player/waves/test/")

for i in dict:
    filename = './gotcha-corona-player/waves/test/' + i
    img = cv.imread(filename)
    print(filename)
    # sizeh = random.randint(480, 1080)
    # scale = sizeh / img.shape[0]
    # print(scale)
    res = cv.resize(img, (512, 512), interpolation=cv.INTER_AREA)
    cv.imwrite(
        './gotcha-corona-player/waves/new-test/' + i, res)
    # cv.imshow("res", res)
    # cv.waitKey(0)
