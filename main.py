#!/usr/bin/env python

# WS server example

# import asyncio
# import websockets


# async def hello(websocket, path):
#     name = await websocket.recv()
#     print(f"< {name}")

#     greeting = f"Hello {name}!"

#     await websocket.send(greeting)
#     print(f"> {greeting}")

# start_server = websockets.serve(hello, "localhost", 80)

# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()

import cv2 as cv
from test import FasterRCNN
import numpy as np

model = FasterRCNN()

img = cv.imread("./Corona/train/10.jpg", cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32)
img /= 255.0

model.eval(img, visuable=True)
