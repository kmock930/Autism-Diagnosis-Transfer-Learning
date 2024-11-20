import random
import math
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

seed = 2042
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

CLIP_LEN = 12         # Number of frames per video clip
RESIZE_HEIGHT = 64    # Frame resize height
CROP_SIZE = 64        # Crop height
SIZE2 = 64            # Crop width

class VideoDataGenerator(Sequence):
    def __init__(self, dataset_paths, labels, batch_size=1, shuffle=True, split='train', augment=False, clip_len=CLIP_LEN):
        self.dataset_paths = dataset_paths  # List of video file paths
        self.labels = labels                # Corresponding video labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.augment = augment
        self.on_epoch_end()
        self.clip_len = clip_len

    def __len__(self):
        return int(np.floor(len(self.dataset_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.dataset_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        X, y = self.__data_generation(batch_paths, batch_labels)
        return X, y

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

        # Check all frames have the same shape
        frame_shapes = [frame.shape for frame in X]
        if not all(shape == frame_shapes[0] for shape in frame_shapes):
            for idx, shape in enumerate(frame_shapes):
                if shape != frame_shapes[0]:
                    print(f"Shape mismatch in batch at index {idx}: {shape} != {frame_shapes[0]}")
            raise ValueError("Not all input arrays have the same shape.")

        X = np.stack(X)
        y = np.array(y)
        return X, y

    def augment_frames(self, frames):
        if random.random() < 0.5:
            frames = [cv2.flip(frame, 1) for frame in frames]
        return frames

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
        frames = self.resize(frames)
        frames = self.normalize(frames)
        frames = self.to_tensor(frames)
        return frames

    def resize(self, frames):
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (RESIZE_HEIGHT, SIZE2), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized)
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

class MultiDatasetDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset1_paths, dataset1_labels, dataset2_paths, dataset2_labels,
                 batch_size=32, shuffle=True, clip_len=12, resize_height=64, resize_width=64, crop_size=64):
        self.dataset1_paths = dataset1_paths
        self.dataset1_labels = dataset1_labels
        self.dataset2_paths = dataset2_paths
        self.dataset2_labels = dataset2_labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Preprocessing parameters
        self.clip_len = clip_len
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.crop_size = crop_size

        self.on_epoch_end()

    def __len__(self):
        # Calculate the number of batches
        return min(len(self.dataset1_paths), len(self.dataset2_paths)) // (self.batch_size // 2)

    def __getitem__(self, index):
        # Retrieve half batch from each dataset
        batch_size_half = self.batch_size // 2

        idx1 = index * batch_size_half
        idx2 = idx1 + batch_size_half
        batch_paths1 = self.dataset1_paths[idx1:idx2]
        batch_labels1 = self.dataset1_labels[idx1:idx2]

        idx3 = index * batch_size_half
        idx4 = idx3 + batch_size_half
        batch_paths2 = self.dataset2_paths[idx3:idx4]
        batch_labels2 = self.dataset2_labels[idx3:idx4]

        # Load data
        X1, y1 = self.load_batch(batch_paths1, batch_labels1)
        X2, y2 = self.load_batch(batch_paths2, batch_labels2)

        # Combine data and labels
        X_batch = np.concatenate([X1, X2], axis=0)
        y_batch = np.concatenate([y1, y2], axis=0)

        # Create dataset identifiers
        dataset_ids1 = np.zeros(len(y1), dtype=np.int32)  # Dataset 1 ID: 0
        dataset_ids2 = np.ones(len(y2), dtype=np.int32)   # Dataset 2 ID: 1
        dataset_ids = np.concatenate([dataset_ids1, dataset_ids2], axis=0)

        return X_batch, y_batch, dataset_ids

    def on_epoch_end(self):
        if self.shuffle:
            # Shuffle each dataset
            self.dataset1_paths, self.dataset1_labels = self._shuffle(self.dataset1_paths, self.dataset1_labels)
            self.dataset2_paths, self.dataset2_labels = self._shuffle(self.dataset2_paths, self.dataset2_labels)

    def _shuffle(self, paths, labels):
        combined = list(zip(paths, labels))
        random.shuffle(combined)
        paths[:], labels[:] = zip(*combined)
        return paths, labels

    def load_batch(self, batch_paths, batch_labels):
        X_batch = []
        y_batch = []

        for path, label in zip(batch_paths, batch_labels):
            frames = self.load_and_preprocess_video(path)
            if frames.shape != (self.clip_len, self.resize_height, self.resize_width, 3):
                print(f"Incorrect frame shape for video {path}: {frames.shape}. Replacing with zeros.")
                frames = np.zeros((self.clip_len, self.resize_height, self.resize_width, 3), dtype=np.float32)
            X_batch.append(frames)
            y_batch.append(label)

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        return X_batch, y_batch

    def load_and_preprocess_video(self, video_path):
        try:
            frames = self.load_frames(video_path)
            frames = self.preprocess_frames(frames)
            if frames.shape != (self.clip_len, self.resize_height, self.resize_width, 3):
                print(f"Preprocessed frames have incorrect shape for {video_path}: {frames.shape}")
                frames = np.zeros((self.clip_len, self.resize_height, self.resize_width, 3), dtype=np.float32)
        except Exception as e:
            print(f"Exception occurred while loading {video_path}: {e}")
            frames = np.zeros((self.clip_len, self.resize_height, self.resize_width, 3), dtype=np.float32)
        return frames

    def load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            cap.release()
            return np.zeros((self.clip_len, self.resize_height, self.resize_width, 3), dtype=np.uint8)

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
            if total_frames < self.clip_len:
                # Read all frames and repeat to reach clip_len
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame is not None:
                        frames.append(frame)
                if len(frames) > 0:
                    repeat_times = math.ceil(self.clip_len / len(frames))
                    frames = frames * repeat_times
                    frames = frames[:self.clip_len]
                else:
                    # If no frames, use blank frames
                    frames = [np.zeros((self.resize_height, self.resize_width, 3), dtype=np.uint8) for _ in range(self.clip_len)]
            else:
                # Uniformly sample clip_len frames
                indices = np.linspace(0, total_frames - 1, self.clip_len).astype(int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        if len(frames) > 0:
                            frame = frames[-1]
                        else:
                            frame = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.uint8)
                    frames.append(frame)

        cap.release()

        # Ensure the number of frames is clip_len
        if len(frames) < self.clip_len:
            if len(frames) > 0:
                last_frame = frames[-1]
            else:
                last_frame = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.uint8)
            while len(frames) < self.clip_len:
                frames.append(last_frame)

        return np.array(frames).astype(np.uint8)

    def preprocess_frames(self, frames):
        # Resize and normalize frames
        frames = self.resize(frames)
        frames = self.normalize(frames)
        return frames

    def resize(self, frames):
        # Resize frames
        resized_frames = []
        for frame in frames:
            try:
                resized = cv2.resize(frame, (self.resize_width, self.resize_height), interpolation=cv2.INTER_LINEAR)
                resized_frames.append(resized)
            except Exception as e:
                print(f"Error resizing frame: {e}. Using zeros frame.")
                resized = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.uint8)
                resized_frames.append(resized)
        return np.array(resized_frames)

    def normalize(self, frames):
        # Normalize to [0, 1]
        return frames.astype(np.float32) / 255.0
