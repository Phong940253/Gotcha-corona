if __name__ == '__main__':
    import asyncio
    import websockets
    import json

    import base64
    import cv2 as cv
    from test import FasterRCNN
    import numpy as np
    import os
    import torch
    torch.multiprocessing.freeze_support()
    model = FasterRCNN()

    # CORONA_TEMPLATE_PATH = os.path.dirname(
    #     os.path.abspath(__file__)) + '/corona_template.png'
    # CORONA_SCALE_RATIO = 0.5

    # corona_template_image = cv2.imread(CORONA_TEMPLATE_PATH, 0)
    # corona_template_image = cv2.resize(
    #     corona_template_image, None, fx=CORONA_SCALE_RATIO, fy=CORONA_SCALE_RATIO)

    # def catch_corona(wave_image, threshold=0.8):
    #     wave_image_gray = cv2.cvtColor(wave_image, cv2.COLOR_BGRA2GRAY)
    #     res = cv2.matchTemplate(
    #         wave_image_gray, corona_template_image, cv2.TM_CCOEFF_NORMED)
    #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    #     if max_val < threshold:
    #         return []

    #     width, height = corona_template_image.shape[::-1]
    #     top_left = max_loc
    #     bottom_right = (top_left[0] + width, top_left[1] + height)

    #     return [[top_left, bottom_right]]

    # def base64_to_image(base64_data):
    #     encoded_data = base64_data.split(',')[1]
    #     nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    #     img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    #     return img

    async def play_game(websocket, path):
        print('Corona Killer is ready to play!')
        catchings = []
        last_round_id = ''
        wave_count = 0
        listImage = []
        listWave = []

        while True:

            # receive a socket message (wave)
            try:
                data = await websocket.recv()
            except Exception as e:
                print('Error: ' + e)
                break

            json_data = json.loads(data)
            listWave.append(json_data["waveId"])

            # check if starting a new round
            if json_data["roundId"] != last_round_id:
                print(f'> Catching corona for round {json_data["roundId"]}...')
                last_round_id = json_data["roundId"]
            base64_data = json_data['base64Image']
            listImage.append(base64_data.split(',')[1])

            # # catch corona in a wave image
            # wave_image = base64_to_image(json_data['base64Image'])
            # results = catch_corona(wave_image)

            # # save result image file for debugging purpose
            # for result in results:
            #     cv2.rectangle(wave_image, result[0], result[1], (0, 0, 255), 2)

            # waves_dir = f'waves/{last_round_id}/'
            # if not os.path.exists(waves_dir):
            #     os.makedirs(waves_dir)

            # cv2.imwrite(os.path.join(
            #     waves_dir, f'{json_data["waveId"]}.jpg'), wave_image)

            # print(f'>>> Wave #{wave_count:03d}: {json_data["waveId"]}')
            wave_count = wave_count + 1

            # # store catching positions in the list
            # catchings.append({
            #     "positions": [
            #         {"x": (result[0][0] + result[1][0]) / 2, "y": (result[0][1] + result[1][1]) / 2} for result in results
            #     ],
            #     "waveId": json_data["waveId"]
            # })

            # send result to websocket if it is the last wave
            if json_data["isLastWave"]:
                round_id = json_data["roundId"]
                boxes, labels, scores = model.eval(
                    listImage, visuable=False, save=True)
                for i, val in enumerate(boxes):
                    listCorona = []
                    for index, box in enumerate(val):
                        # neu ti le cao thi bat
                        if scores[i][index] > 0.35 and labels[i][index] == 2:
                            inCircle = False
                            box = box * (800 / 512)
                            box = box.astype(int)
                            # print(type(box), box)
                            x_min, y_min, x_max, y_max = box
                            x, y = (x_max + x_min) / 2, (y_max + y_min) / 2
                            for indpeo, people in enumerate(val):
                                people = people * (800 / 512)
                                people = people.astype(int)
                                if scores[i][indpeo] > 0.35 and labels[i][indpeo] == 1:
                                    x_min_peo, y_min_peo, x_max_peo, y_max_peo = people
                                    center_x, center_y = (
                                        x_max_peo + x_min_peo) / 2, (y_max_peo + y_min_peo) / 2
                                    radius = max(
                                        x_max_peo - x_min_peo, y_max_peo - y_min_peo) / 2
                                    if (x-center_x)**2 + (y-center_y)**2 <= radius**2:
                                        inCircle = True
                                        break
                            if not inCircle:
                                listCorona.append({"x": x, "y": y})
                    catchings.append({
                        "positions": listCorona,
                        "waveId": listWave[i]
                    })

                print(f'> Submitting result for round {round_id}...')

                json_result = {
                    "roundId": round_id,
                    "catchings": catchings,
                }
                with open('./Corona/json/test.json', 'w') as f:
                    json.dump(catchings, f)

                await websocket.send(json.dumps(json_result))
                print('> Submitted.')

                catchings = []
                wave_count = 0
                listImage = []
                listWave = []

    # print("ready to start!")
    start_server = websockets.serve(
        play_game, "localhost", 8765, max_size=100000000)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

    # PATH_TEST = "./Corona/train/"

    # listImage = []

    # # test image as base 64
    # for i in range(1000):
    #     train_filename = np.random.choice(os.listdir(PATH_TEST))
    #     data = open(PATH_TEST + train_filename, "rb").read()
    #     encoded = base64.b64encode(data)
    #     listImage.append(encoded)
