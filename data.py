import traceback;
import os;
from zipfile import ZipFile;
import constants;
import numpy as np;
import cv2;
import gc;
import random;

def extractZIP(dataset: str = constants.TIKHARM_DATASET, dataset_alias: str = ""):
    '''
    Extract an archive of a dataset into a directory.

    Parameters
    ------------------
    dataset : str - File name of the dataset
    dataset_alias : str (optional) - Alias of the dataset (for the name of the root directory)

    Returns
    ------------------
    bool : whether the extraction is successful or not
    '''
    try:
        extract_path = os.path.join(constants.ROOT_DIR, dataset_alias if dataset_alias != "" else dataset);
        os.makedirs(extract_path, exist_ok=True);
        with ZipFile(dataset + '.zip', 'r') as datasetZIP:
            datasetZIP.extractall(  );
        return True;
    except Exception as e:
        traceback.print_exception(e);
        return False;

def load_video(path: str, dataset: str = constants.TIKHARM_DATASET, toDisplay: bool = False):
    '''
    Load a Video from a given full filepath into a Numpy array representation, 
    and optionally, 
    Visualize the video.

    Parameters
    ------------------------
    path : str - A complete filepath of the video to be loaded.
    dataset : str - Name of the dataset which the video belongs to.
    toDisplay : bool - Whether to visualize the video or not. 

    Returns
    ------------------------
    Numpy.ndarray : the numeric representation of the video.
    '''
    capture = cv2.VideoCapture(path);

    # Get the total number of frames in the video
    # https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT));

    # FRAME SELECTION: 
    # Based on the paper, use the same ratio to select frames uniformly
    # they used 16 frames in the first dataset out of 21-30 frames per video,
    # and 10 frames in the second dataset out of 12-20 frames
    if (dataset == constants.TIKHARM_DATASET):
        # 1st dataset in our transfer learning
        num_sample_frames: int = round((16 / np.mean([21,30])) * num_frames);
    elif (dataset == constants.VIOLENCE_DATASET):
        num_sample_frames: int = round((10 / np.mean([12,20])) * num_frames);
    frame_indices = np.linspace(0, num_frames - 1, num_sample_frames, dtype=int)

    keyFrames: list = [];
    for i in range(num_frames):
        ret, frame = capture.read();
        if (ret != True):
            break;
        
        # Visualize a random video from the set
        if (toDisplay == True):
            cv2.imshow('Raw Video', frame);
            if (cv2.waitKey(30) & 0xFF == ord('q')):
                break;

        frame = preprocessFrame(frame);
        
        if (i in frame_indices):
            keyFrames.append(frame);
        
        # Visualize a random video from the set
        if (toDisplay == True):
            cv2.imshow('Preprocessed Video', frame);
            if (cv2.waitKey(30) & 0xFF == ord('q')):
                break;
    
    # https://stackoverflow.com/questions/65446464/how-to-convert-a-video-in-numpy-array
    video = np.stack(keyFrames, axis=0);

    # cleanup
    capture.release();
    cv2.destroyAllWindows();

    return video;

def preprocessFrame(frame):
    # Resize the frame
    frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC);
    # Make the frame grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    # Adaptive Thresholding
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2);
    return frame;

def read_videos(dataset: str = constants.TIKHARM_DATASET):
    '''
    Read Videos from extracted directories, by uniformly select key frames from each video.
    
    Parameters
    ------------------------
    dataset: str - Name of the dataset

    Returns
    ------------------------
    X_train : numpy.ndarray
    y_train : numpy.ndarray
    X_val : numpy.ndarray
    y_val : numpy.ndarray
    X_test : numpy.ndarray
    y_test : numpy.ndarray
    '''
    X_train: list = [];
    X_val: list = [];
    X_test: list = [];
    y_train: list = [];
    y_val: list = [];
    y_test: list = [];

    directory = os.path.join(constants.ROOT_DIR, dataset);
    for rootName in os.listdir(dataset):
        currPath = os.path.join(directory, rootName);
        if (os.path.isdir(currPath) == True):
            originalSetSize: int = 0;
            selectedSetSize: int = 0;
            for dirName in os.listdir(currPath):
                currPath = os.path.join(currPath, dirName);
                if (os.path.isdir(currPath) == True and dirName in ['train', 'test', 'val']):
                    for categoryName in os.listdir(currPath):
                        currPath = os.path.join(currPath, categoryName);
                        # iterate through video files
                        videosList: list[str] = os.listdir(currPath);
                        # narrow down the dataset
                        selected_videosList: list[str] = random.sample(videosList, k=round(len(videosList)/3));
                        originalSetSize += len(videosList);
                        selectedSetSize += len(selected_videosList);
                        # randomize a video for visualization
                        randInd: int = random.randint(0, len(selected_videosList)-1);
                        randVid_name = selected_videosList[randInd];
                        for videoName in selected_videosList:
                            currPath = os.path.join(currPath, videoName);
                            if (os.path.isfile(currPath) and videoName.endswith(constants.ALLOWED_FILE_FORMATS)):
                                # Save Videos
                                video = load_video(
                                    path=currPath,
                                    dataset=dataset,
                                    toDisplay=videoName == randVid_name
                                );
                                # Map the category names and samples into array
                                match (dirName):
                                    case 'train':
                                        X_train.append(video);
                                        y_train.append(categoryName);
                                    case 'test':
                                        X_test.append(video);
                                        y_test.append(categoryName);
                                    case 'val':
                                        X_val.append(video);
                                        y_val.append(categoryName);
                                # Visualize shape
                                if (videoName == randVid_name):
                                    print(f"Shape of a video of {categoryName} category in the {dirName} set in {dataset} dataset: {video.shape}.");
                                
                            else:
                                raise FileExistsError(f"Unexpected File/Directory {currPath} exists at the end level in your {dataset} dataset.");
                            # reset currPath for next iteration
                            currPath = currPath.replace(videoName, "");
                        # reset currPath for next iteration
                        currPath = currPath.replace(categoryName, "");
                else:
                    raise FileExistsError(f"Unexpected File/Directory '{currPath}' exists at 2nd level in your {dataset} dataset.");
                # reset currPath for next iteration
                currPath = currPath.replace(dirName, "");
            print(f"Size of Original {dirName} set: {originalSetSize} ; Size of truncated {dirName} set: {selectedSetSize}.");
            # reset set size for next dataset
            originalSetSize = 0;
            selectedSetSize= 0;
            # free up memory
            del videosList, selected_videosList;
            gc.collect();
        else:
            raise FileExistsError(f"Unexpected File exists at root level in your {dataset} dataset.");

    X_train = np.array(X_train);
    y_train = np.array(y_train);
    X_val = np.array(X_val);
    y_val = np.array(y_val);
    X_test = np.array(X_test);
    y_test = np.array(y_test);

    return X_train, y_train, X_val, y_val, X_test, y_test;
