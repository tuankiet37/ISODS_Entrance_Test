import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import time
from tqdm import trange
import torch

def main(video_path):
    video_path = video_path.replace('\\', '/')
    video_name = os.path.basename(video_path).split('.')[0]

    directory = os.getcwd()
    save_path = os.path.join(directory, "Output", video_name, "The truck with color property")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    model = YOLO("./Model/yolov8n.pt")


    color_ranges = {
        "red": ([0, 100, 20], [10, 255, 255]),
        "green": ([35, 100, 20], [85, 255, 255]),
        "blue": ([100, 100, 20], [140, 255, 255]),
        "white": ([0, 0, 200], [180, 30, 255]),
        "yellow": ([20, 100, 20], [35, 255, 255]),
    }

    for index in trange(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, verbose=False, classes = 7)[0]

        if len(results) != 0:

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                delta_y = (y2 - y1) // 8
                delta_x = (x2 - x1) // 8
                truck = frame[int(y1 + delta_y):int(y2 - delta_y), int(x1 + delta_x):int(x2 - delta_x), :]

                hsv = cv2.cvtColor(truck, cv2.COLOR_BGR2HSV)

                max_color = ""
                max_count = 0

                for color, (lower, upper) in color_ranges.items():
                    lower_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    count = np.count_nonzero(lower_mask)

                    if count > max_count:
                        max_count = count
                        max_color = color

                text = "the " + max_color + " truck" if max_color else "a truck"
                cv2.putText(frame, text, (int(x1), int(y1 + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 225), thickness=2)
                cv2.imwrite(os.path.join(save_path, "frame_" + str(index) + '.jpg'), frame)
    cap.release()

if __name__ == '__main__':
    print("Torch cuda available: ", torch.cuda.is_available())
    video_path = sys.argv[1]
    start_time = time.time()
    main(video_path)
    elapsed_minutes = (time.time() - start_time) / 60
    rounded_elapsed_minutes = round(elapsed_minutes, 3)
    print("--- %.3f minutes ---" % rounded_elapsed_minutes)
