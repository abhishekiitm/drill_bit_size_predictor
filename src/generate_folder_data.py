import os

import cv2
import imagehash
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from PIL import Image


def get_train_metadata(video_filename):
    width = int(video_filename[:2])
    height = int(video_filename[3:5])
    is_dark = True if video_filename[5:9] == "dark" else False
    return width, height, is_dark


class DataGenerator(object):
    def __init__(
        self,
        raw_input_dir,
        output_dir,
        SAMPLE_TIME_SEC,
        CROP_X,
        CROP_Y,
        CROP_H,
        CROP_W,
        hash_size,
        highfreq_factor,
    ) -> None:
        self.raw_input_dir = raw_input_dir
        self.output_dir = output_dir
        self.SAMPLE_TIME_SEC = SAMPLE_TIME_SEC
        self.CROP_X = CROP_X
        self.CROP_Y = CROP_Y
        self.CROP_H = CROP_H
        self.CROP_W = CROP_W
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    def generate_data_from_video(self, video_filename):
        video_path = os.path.join(self.raw_input_dir, video_filename)
        cap = cv2.VideoCapture(video_path)

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        sample_frame_no = max(int(fps_video * self.SAMPLE_TIME_SEC), 1)

        frames_counter = 0

        while True:
            check, frame = cap.read()
            frames_counter = frames_counter + 1
            if frames_counter % sample_frame_no != 0:
                continue

            if not check:
                print(
                    f"Video: {video_filename} processed, no of image data generated: "
                )
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            output_filename = (
                f"{video_filename[:-4]}_{str(frames_counter).rjust(5,'0')}.png"
            )
            output_path = os.path.join(self.output_dir, output_filename)
            image.save(output_path, bitmap_format="png")

        cap.release()

    def generate_data(self):
        video_filenames = [
            x for x in os.listdir(self.raw_input_dir) if x.endswith(".mp4")
        ]

        self.image_paths = []
        self.widths = []
        self.heights = []
        self.is_dark = []
        self.frame_nos = []
        # idx = 10
        for video_filename in video_filenames:
            self.generate_data_from_video(video_filename)

        zipped_cols = list(
            zip(
                self.image_paths,
                self.widths,
                self.heights,
                self.is_dark,
                self.frame_nos,
            )
        )
        col_names = ["image_path", "width", "height", "is_dark", "frame_number"]
        data_df = pd.DataFrame(zipped_cols, columns=col_names)
        data_df.to_csv(os.path.join(self.output_dir, "data.csv"))


if __name__ == "__main__":
    SAMPLE_TIME_SEC = 10
    CROP_X = 390
    CROP_Y = 475
    CROP_W = 140
    CROP_H = 40
    hash_size = 3
    highfreq_factor = 2

    raw_input_dir = "data_raw_videos"
    output_dir = "test_folder"

    data_generator = DataGenerator(
        raw_input_dir,
        output_dir,
        SAMPLE_TIME_SEC,
        CROP_X,
        CROP_Y,
        CROP_H,
        CROP_W,
        hash_size,
        highfreq_factor,
    )
    data_generator.generate_data()
