from ultralytics import YOLO
import os
import cv2
from tqdm import trange
import torch
import sys
import time
import numpy as np

def main(video_path):
    video_path = video_path.replace('\\', '/')
    video_name = os.path.basename(video_path).split('.')[0]

    directory = os.getcwd()
    save_path = os.path.join(directory, "Output", video_name, "The truck")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    model = YOLO("./Model/yolov8n.pt")

    for index in trange(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, verbose=False, classes = 7)[0]

        if len(results) != 0:
            annotated_frame = results.plot()
            cv2.imwrite(os.path.join(save_path, "frame_" + str(index) + '.jpg'), annotated_frame)

if __name__ == '__main__':
    print("Torch cuda available: ", torch.cuda.is_available())
    video_path = sys.argv[1]
    start_time = time.time()
    main(video_path)
    elapsed_minutes = (time.time() - start_time) / 60
    rounded_elapsed_minutes = round(elapsed_minutes, 3)
    print("--- %.3f minutes ---" % rounded_elapsed_minutes)
