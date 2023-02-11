"""
This module is used to generate training data from the videos
"""
import os

import cv2
import imagehash
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from PIL import Image


def get_train_metadata(video_filename):
    """
    extracts relevant metadata from the video filename such as width,
    height and if the video was taken in a dark environment
    """
    width = int(video_filename[:2])
    height = int(video_filename[3:5])
    is_dark = True if video_filename[5:9] == "dark" else False
    return width, height, is_dark


class DataGenerator(object):
    """
    generates data by processing all the videos in the raw_input_dir directory
    """

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
        """
        :raw_input_dir: folder location that has the videos
        :output_dir: folder where the generated images will be saved
        :SAMPLE_TIME_SEC: sample frame every SAMPLE_TIME_SEC seconds
        :CROP_X: coordinate where to crop image
        :CROP_Y: coordinate where to crop image
        :CROP_H: height of cropped image
        :CROP_W: width of cropped image
        :hash_size: hash size that will be used for perceptual hashing
        :highfreq_factor: high frequency factor that will be used for perceptual hashing
        """
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

        # every sample_frame_no'th frame is sampled
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        sample_frame_no = max(int(fps_video * self.SAMPLE_TIME_SEC), 1)

        width, height, is_dark = get_train_metadata(video_filename)

        frames_counter = 0

        # dictionary to store the image hashes created by perceptual hashing
        hashed_images = {}

        while True:
            check, frame = cap.read()
            frames_counter = frames_counter + 1

            # only every sample_frame_no'th frame is sampled
            if frames_counter % sample_frame_no != 0:
                continue

            # handles the case when we have reached the end of video
            if not check:
                print(
                    f"Video: {video_filename} processed, no of image data generated: {len(hashed_images)}"
                )
                break

            # frame processing is done here

            # cropped to only get the center part which is relevant to us for pattern recognition
            cropped_frame = frame[Y : Y + H, X : X + W]

            # convert to grayscale, I am discarding colour information for faster processing
            # also visually grayscale seems to have enough information for our task
            gray_img = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

            # perceptual hashing needs image in PIL Image format
            img = Image.fromarray(gray_img)
            # img.show()

            # every sampled image's hash is calculated
            # image is saved only if its hash is not in the previous hashes (hashed_images)
            # this is because similar images will generate the same hash
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
        """
        method that initiates data generation

        metadata related to the generate images is store in {output_dir}/data.csv
        """
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

        # metadata related to the generate images is store in {output_dir}/data.csv
        data_df = pd.DataFrame(zipped_cols, columns=col_names)
        data_df.to_csv(os.path.join(self.output_dir, "data.csv"))


if __name__ == "__main__":
    # sample video every SAMPLE_TIME_SEC seconds
    SAMPLE_TIME_SEC = 0.1

    # cropping details
    CROP_X = 390
    CROP_Y = 475
    CROP_W = 140
    CROP_H = 40

    # Since frames closer in time will be very similar to each other I have
    # used perceptual hashing to identify frames that are sufficiently different
    # perceptual hashing generates a fingerprint for each sampled frame which can be
    # compared against for similar frames
    #
    # Parameters for perceptual hashing were chosen experimentally
    hash_size = 3
    highfreq_factor = 2

    # folder location that has the videos
    raw_input_dir = "data_raw_videos"

    # folder where the generated images will be saved
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
