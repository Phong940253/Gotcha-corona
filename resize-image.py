import os
import cv2 as cv
import random

dict = os.listdir("./Corona/Not-resize/")

for i in dict:
    filename = './Corona/Not-resize/' + i
    img = cv.imread(filename)
    print(filename)
    sizeh = random.randint(480, 1080)
    scale = sizeh / img.shape[0]
    print(scale)
    res = cv.resize(img, None, fx=scale, fy=scale,
                    interpolation=cv.INTER_AREA)
    cv.imwrite('./Corona/background/' + i, res)
    # cv.imshow("res", res)
    # cv.waitKey(0)
