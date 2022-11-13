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
        X = self.CROP_X
        Y = self.CROP_Y
        H = self.CROP_H
        W = self.CROP_W

        hash_size = self.hash_size
        highfreq_factor = self.highfreq_factor

        video_path = os.path.join(self.raw_input_dir, video_filename)
        cap = cv2.VideoCapture(video_path)

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        sample_frame_no = max(int(fps_video * self.SAMPLE_TIME_SEC), 1)

        width, height, is_dark = get_train_metadata(video_filename)
        frames_counter = 0

        hashed_images = {}

        while True:
            check, frame = cap.read()
            frames_counter = frames_counter + 1
            if frames_counter % sample_frame_no != 0:
                continue

            if not check:
                print(
                    f"Video: {video_filename} processed, no of image data generated: {len(hashed_images)}"
                )
                break

            cropped_frame = frame[Y : Y + H, X : X + W]
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cropped_frame)
            # img.show()

            hashed_image = imagehash.phash(
                img, hash_size=hash_size, highfreq_factor=highfreq_factor
            )

            if hashed_image not in hashed_images:
                output_filename = (
                    f"{video_filename[:-4]}_{str(frames_counter).rjust(5,'0')}.png"
                )
                output_path = os.path.join(self.output_dir, output_filename)
                img.save(output_path, bitmap_format="png")
                self.image_paths.append(output_path)
                self.widths.append(width)
                self.heights.append(height)
                self.is_dark.append(is_dark)
                self.frame_nos.append(frames_counter)

            hashed_images[hashed_image] = hashed_images.get(hashed_image, []) + [img]

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
    SAMPLE_TIME_SEC = 0.1
    CROP_X = 380
    CROP_Y = 455
    CROP_H = 80
    CROP_W = 170
    hash_size = 3
    highfreq_factor = 2

    raw_input_dir = "data_raw_videos"
    output_dir = "data_generated_images"

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
