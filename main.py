if __name__ == '__main__':
    import base64
    import cv2 as cv
    from test import FasterRCNN
    import numpy as np
    import os
    import torch
    torch.multiprocessing.freeze_support()

    model = FasterRCNN()
    PATH_TEST = "./Corona/train/"

    listImage = []

    # test image as base 64
    for i in range(1000):
        train_filename = np.random.choice(os.listdir(PATH_TEST))
        data = open(PATH_TEST + train_filename, "rb").read()
        encoded = base64.b64encode(data)
        listImage.append(encoded)

    model.eval(listImage, visuable=False, save=False)
