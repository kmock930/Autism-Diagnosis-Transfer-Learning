"""Adapted from MDSupCL/data/data_uniform_Sup.py (https://github.com/asharani97/MDSupCL/blob/main/data/data_uniform_Sup.py)"""
import random
import math
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter
from tensorflow.keras.utils import Sequence

seed = 2042
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

CLIP_LEN = 12         # Number of frames per video clip
RESIZE_HEIGHT = 64    # Frame resize height
CROP_SIZE = 64        # Crop height
SIZE2 = 64            # Crop width

class SSCLVideoDataGenerator(Sequence):
    def __init__(self, dataset_paths, labels, batch_size=1, shuffle=True, split='train', augment=True, double_view=True):
        self.indexes = None
        self.dataset_paths = dataset_paths  # List of video file paths
        self.labels = labels                # Corresponding video labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.on_epoch_end()
        self.augment = augment
        self.double_view = double_view

    def __len__(self):
        return int(np.floor(len(self.dataset_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        video_paths_temp = [self.dataset_paths[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]

        X1_batch = []
        X2_batch = []
        y_batch = []

        for i in range(len(video_paths_temp)):
            video_path = video_paths_temp[i]
            label = labels_temp[i]

            frames = self.load_frames(video_path)

            # Generate augmented views
            frames_aug1 = self.augment_frames(frames)
            frames_aug2 = None
            if self.double_view:
                frames_aug2 = self.augment_frames(frames)

            # Preprocess frames
            frames_aug1 = self.preprocess_frames(frames_aug1)
            if self.double_view:
                frames_aug2 = self.preprocess_frames(frames_aug2)

            X1_batch.append(frames_aug1)
            if self.double_view:
                X2_batch.append(frames_aug2)
            y_batch.append(label)

        X1_batch = np.array(X1_batch)
        y_batch = np.array(y_batch)

        if self.double_view:
            X2_batch = np.array(X2_batch)
            return (X1_batch, X2_batch), y_batch
        else:
            return X1_batch, y_batch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataset_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_paths, batch_labels):
        X = []
        y = []
        for i, video_path in enumerate(batch_paths):
            frames = self.load_frames(video_path)
            if self.split == 'train' and self.augment:
                # Generate two augmented versions
                frames_1 = self.augment_frames(frames)
                frames_2 = self.augment_frames(frames)
                frames_1 = self.preprocess_frames(frames_1)
                frames_2 = self.preprocess_frames(frames_2)
                X.append(frames_1)
                y.append(batch_labels[i])
                X.append(frames_2)
                y.append(batch_labels[i])
            else:
                frames = self.preprocess_frames(frames)
                X.append(frames)
                y.append(batch_labels[i])
        X = np.array(X)
        y = np.array(y)
        return X, y

    def augment_frames(self, frames):
        augmented_frames = []
        for frame in frames:
            # Random color jitter
            frame = self.color_jitter(frame)
            # Random grayscale
            if random.random() < 0.2:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Random Gaussian blur
            if random.random() < 0.5:
                frame = cv2.GaussianBlur(frame, (5, 5), 0)
            # Random horizontal flip
            if random.random() < 0.5:
                frame = cv2.flip(frame, 1)
            augmented_frames.append(frame)
        return np.array(augmented_frames)

    def load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            # Return zero frames to avoid shape inconsistency
            return np.zeros((CLIP_LEN, RESIZE_HEIGHT, SIZE2, 3), dtype=np.uint8)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total_frames <= 0:
            # Handle case when total frames are not available
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is not None:
                    frames.append(frame)
        else:
            if total_frames < CLIP_LEN:
                # Read all frames and repeat to reach CLIP_LEN
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame is not None:
                        frames.append(frame)
                if len(frames) > 0:
                    repeat_times = math.ceil(CLIP_LEN / len(frames))
                    frames = frames * repeat_times
                    frames = frames[:CLIP_LEN]
                else:
                    # If no frames, use blank frames
                    frames = [np.zeros((1024, 576, 3), dtype=np.uint8) for _ in range(CLIP_LEN)]
            else:
                # Uniformly sample CLIP_LEN frames
                indices = np.linspace(0, total_frames - 1, CLIP_LEN).astype(int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # If cannot read, use last frame or blank frame
                        if len(frames) > 0:
                            frame = frames[-1]
                        else:
                            frame = np.zeros((1024, 576, 3), dtype=np.uint8)
                    frames.append(frame)

        cap.release()

        # Ensure the number of frames is CLIP_LEN
        if len(frames) < CLIP_LEN:
            if len(frames) > 0:
                last_frame = frames[-1]
            else:
                last_frame = np.zeros((1024, 576, 3), dtype=np.uint8)
            while len(frames) < CLIP_LEN:
                frames.append(last_frame)

        return np.array(frames).astype(np.uint8)

    def preprocess_frames(self, frames):
        # Includes resize, normalize, to_tensor, etc.
        frames = self.resize(frames)
        frames = self.normalize(frames)
        frames = self.to_tensor(frames)
        return frames

    def resize(self, frames):
        resized_frames = []
        for frame in frames:
            frame = cv2.resize(frame, (RESIZE_HEIGHT, RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(frame)
        return np.array(resized_frames)

    def normalize(self, frames):
        return frames.astype(np.float32) / 255.0

    def to_tensor(self, frames):
        return frames

    def random_flip(self, frames):
        if random.random() < 0.5:
            frames = frames[:, :, ::-1, :]
        return frames

    def crop(self, frames, clip_len, crop_size, crop_size2):
        if frames.shape[0] > clip_len:
            time_index = random.randint(0, frames.shape[0] - clip_len)
        else:
            time_index = 0
        height_index = random.randint(0, frames.shape[1] - crop_size)
        width_index = random.randint(0, frames.shape[2] - crop_size2)
        frames = frames[time_index:time_index + clip_len, height_index:height_index + crop_size,
                        width_index:width_index + crop_size2, :]
        if frames.shape[0] < clip_len:
            pad_num = clip_len - frames.shape[0]
            frames = np.concatenate((frames, frames[:pad_num]), axis=0)
        return frames

    def color_jitter(self, frame):
        # Convert to PIL image
        img = Image.fromarray(frame)

        # Randomly change brightness
        if random.random() < 0.8:
            brightness_factor = random.uniform(0.6, 1.4)
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        # Randomly change contrast
        if random.random() < 0.8:
            contrast_factor = random.uniform(0.6, 1.4)
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        # Randomly change saturation
        if random.random() < 0.8:
            saturation_factor = random.uniform(0.6, 1.4)
            img = ImageEnhance.Color(img).enhance(saturation_factor)

        # Randomly change hue
        if random.random() < 0.8:
            hue_factor = random.uniform(-0.1, 0.1)
            img = np.array(img.convert('HSV'))
            img[:, :, 0] = (img[:, :, 0].astype(int) + int(hue_factor * 255)) % 255
            img = Image.fromarray(img, mode='HSV').convert('RGB')

        # Convert back to NumPy array
        frame = np.array(img)

        return frame

    def gaussian_blur(self, frame):
        # Convert to PIL image
        img = Image.fromarray(frame)

        # Apply Gaussian blur with certain probability
        if random.random() < 0.5:
            radius = random.uniform(0.1, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius))

        # Convert back to NumPy array
        frame = np.array(img)

        return frame
