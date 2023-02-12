import cv2
import math
import numpy as np
import pathlib
import pandas as pd
import random
import imageio
from IPython.display import Image

if __name__ == "__main__":
    DEBUG = False

# Returns the cv2.VideoCapture handle. Remember to release the handle once you are done.
def open_video(video_filepath):
    vid_cap = cv2.VideoCapture(video_filepath)
    
    if not vid_cap.isOpened():
        raise Exception(f'Error opening video {video_filepath}')
        
    return vid_cap

def get_event_frame_index(video_filepath, event_timestamp_millis):
    vid_cap = open_video(video_filepath)
    vid_fps = vid_cap.get(cv2.CAP_PROP_FPS)
    vid_cap.release()
    
    return math.floor(event_timestamp_millis / 1000 * vid_fps) # todo: revisit this if there is issue with indexing too early

def get_frame_indexes_surrounding_event(video_filepath, event_timestamp_millis, sequence_length, frame_step):
    vid_cap = open_video(video_filepath)
    vid_frame_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_cap.release()
    
    event_frame_index = get_event_frame_index(video_filepath, event_timestamp_millis)
    
    max_frame_steps_event_to_beginning = math.floor(event_frame_index / frame_step)
    max_frame_steps_event_to_end = math.floor((vid_frame_count - 1 - event_frame_index)/ frame_step) # reason for minus 1 is this is about index and not about frame count
    
    max_possible_sequence_length = max_frame_steps_event_to_end + max_frame_steps_event_to_beginning + 1
    
    # plus 1 to include the event frame itself
    if sequence_length > max_possible_sequence_length:
        raise Exception(f"Not possible for frame step {frame_step} and sequence length {sequence_length}. Maximum possible sequence length is {max_possible_sequence_length}")

    # Min and max start frame in which the event frame is still included at the exact point
    # considering the requested sequence length and frame step size.
    min_start_frame_idx = event_frame_index - frame_step * min(sequence_length - 1, max_frame_steps_event_to_beginning)
    max_start_frame_idx = event_frame_index - frame_step * max(0, sequence_length - 1 - max_frame_steps_event_to_end)

    frame_indexes = []
    
    for start_frame_idx in range(min_start_frame_idx, (max_start_frame_idx + frame_step), frame_step):
        # generate the index
        frame_indexes.append([i for i in range(start_frame_idx, start_frame_idx + sequence_length * frame_step, frame_step)])

    frame_indexes = np.array(frame_indexes)
    labels = np.array(frame_indexes >= event_frame_index, dtype=np.int8)
    
    return frame_indexes, labels

# test
if __name__ == "__main__":
    video_filepath = 'data/hand_collision_videos/hand_collision.mp4'

    sample_result = get_frame_indexes_surrounding_event(video_filepath, event_timestamp_millis=1000, sequence_length=6, frame_step=13)
    assert sample_result[0].shape == (3, 6)

    if DEBUG:
        print(f'Data shape: {sample_result[0].shape}')
        print(f'Label shape: {sample_result[1].shape}')
        print(sample_result)

    try:
        get_frame_indexes_surrounding_event(video_filepath, event_timestamp_millis=1000, sequence_length=14, frame_step=13)
        assert False # shouldn't get to this code as we expect exception to be thrown
    except Exception as exc:
        if DEBUG:
            print(f'Received expected exception with message "{exc}".')

    sample_result = get_frame_indexes_surrounding_event(video_filepath, event_timestamp_millis=1000, sequence_length=8, frame_step=13)
    assert sample_result is not None

    if DEBUG:
        print(f'Data shape: {sample_result[0].shape}')
        print(f'Label shape: {sample_result[1].shape}')
        print(sample_result)

    print('All tests OK.')

# This function will get image frames from the given video file path,
# for the requested image frame indexes.
# You can request several image frame index sequences.
# Each row in the frame index sequence array corresponds to each sequence.
# The returned image frames will be in form of numpy,
# The numpy array will be arranged following the requested image frame index sequences.
# Returned images will be in RGB format.
# The numpy arrays for the image frames are read-only.
# The same image frame will share the same memory location, even though they appear in multiple sequences.
# The input argument `frame_index_sequences` should be a list of list, e.g. [[2,3,4], [1,2,3]]
# The returned value will be a python list instead of a numpy, to cater for the case of non-homogeneous array.
def get_image_frames(video_filepath, frame_index_sequences, format_frame_fn=None):
    vid_cap = open_video(video_filepath)
    
    unique_frame_indexes = set([idx for sequence in frame_index_sequences for idx in sequence])
    
    frames_map = {}
    
    for frame_idx in unique_frame_indexes:
        ret = vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        if not ret:
            raise Exception(f'Failed to set the frame position for the VideoCapture.')
        
        ret, frame = vid_cap.read()
        if not ret:
            raise Exception(f'Failed to read image frame index {frame_idx}.')
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = np.array(frame)
        
        if format_frame_fn is not None:
            frame = format_frame_fn(frame)
        
        # Set the numpy array to be read only, because we want this same array to be referenced
        # in multiple location in the returned array.
        frame.flags.writeable = False
        
        frames_map[frame_idx] = frame
        
    vid_cap.release()
    
    vid_frames = [np.array([frames_map[frame_idx] for frame_idx in sequence]) for sequence in frame_index_sequences]
    
    return vid_frames

# test
if __name__ == "__main__":
    video_filepath = 'data/hand_collision_videos/hand_collision.mp4'

    result = get_image_frames(video_filepath, [[1,2,3], [2,3,4,5]])
    assert result[0].shape == (3, 720, 1280, 3)
    assert result[1].shape == (4, 720, 1280, 3)

    if DEBUG:
        print(result)

    print('All tests OK.')

    
def get_index_proportion_of_ones(labels, min_proportion_of_ones, max_proportion_of_ones):
    proportion_of_ones = np.sum(labels, axis=-1) / labels.shape[-1]
    
    return (min_proportion_of_ones <= proportion_of_ones) & (proportion_of_ones <= max_proportion_of_ones)

# test
if __name__ == "__main__":
    video_filepath = 'data/hand_collision_videos/hand_collision.mp4'
    frame_index_sequences, labels = get_frame_indexes_surrounding_event(video_filepath, event_timestamp_millis=1500, sequence_length=7, frame_step=10)
    indexes = get_index_proportion_of_ones(labels, 0.5, 0.8)

    assert frame_index_sequences[indexes].shape == (2, 7)
    assert labels[indexes].shape == (2, 7)

    if DEBUG:
        print(frame_index_sequences[indexes])
        print(labels[indexes])

    print('All tests OK.')
    
def get_frames_surrounding_event(video_filepath, event_timestamp_millis, sequence_length, frame_step, min_proportion_of_after_event_frames, max_proportion_of_after_event_frames, format_frame_fn=None):
    frame_idxs, labels = get_frame_indexes_surrounding_event(video_filepath, event_timestamp_millis, sequence_length, frame_step)
    filter_idxs = get_index_proportion_of_ones(labels, min_proportion_of_after_event_frames, max_proportion_of_after_event_frames)
    frame_idxs = frame_idxs[filter_idxs]
    labels = labels[filter_idxs]
    frame_imgs = get_image_frames(video_filepath, frame_idxs, format_frame_fn=format_frame_fn)
    frame_imgs = np.array(frame_imgs) # convert to numpy array since the shape is homogeneous
    
    return frame_imgs, labels

# test
if __name__ == "__main__":
    video_filepath = 'data/hand_collision_videos/hand_collision.mp4'
    frame_imgs, labels = get_frames_surrounding_event(video_filepath,
                                                      event_timestamp_millis=1500,
                                                      sequence_length=15,
                                                      frame_step=5,
                                                      min_proportion_of_after_event_frames=0.4,
                                                      max_proportion_of_after_event_frames=0.6)

    assert frame_imgs.shape == (3, 15, 720, 1280, 3)
    assert labels.shape == (3, 15)

    if DEBUG:
        print('frame_imgs.shape:')
        print(frame_imgs.shape)
        print('labels.shape:')
        print(labels.shape)
        print('labels:')
        print(labels)

    print('All tests OK.')

def get_random_sequence(seq_length, step_size, total_frames):
    total_seq_frames = step_size * seq_length

    if total_seq_frames > total_frames:
        raise Exception(f'Number of video frames ({total_frames}) is not enough for the requested total sequence frames ({total_seq_frames})')

    max_allowed_offset = total_frames - total_seq_frames

    start_index = random.randint(0, max_allowed_offset)
    end_index = start_index + seq_length * step_size

    return list(range(start_index, end_index, step_size))

# test
if __name__ == "__main__":
    # test that no error happens after 100000 sampling of random sequences
    for i in range(100000):
        seq_length = 13
        result = get_random_sequence(seq_length=seq_length, step_size=33, total_frames=774)
        assert len(result) == seq_length

    print('All tests OK.')

def get_random_video_frames(video_filepath, seq_length, step_size, num_seqs=1, format_frame_fn=None):
    vid_cap = open_video(video_filepath)
    vid_frame_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_cap.release()
    
    seq_indexes = []
    for i in range(num_seqs):
        seq_indexes.append(get_random_sequence(seq_length=seq_length, step_size=step_size, total_frames=vid_frame_count))
    
    # return as numpy array because we are sure that the array is homogeneous
    return np.array(get_image_frames(video_filepath, seq_indexes, format_frame_fn))

# test
if __name__ == "__main__":
    video_filepath = 'data/hand_collision_videos/hand_collision.mp4'
    num_seqs = 4

    result = get_random_video_frames(video_filepath=video_filepath, seq_length=13, step_size=5, num_seqs=num_seqs)
    assert result.shape == (num_seqs, 13, 720, 1280, 3)

    print('All tests OK.')

class FrameGenerator:
    def __init__(self, videos_dir_path, labels_file_path, sequence_length, frame_step_size,
                 min_proportion_of_after_event_frames, max_proportion_of_after_event_frames, num_sequences_for_no_event_videos,
                 format_frame_fn=None):
        self.videos_dir_path = videos_dir_path
        self.labels_file_path = labels_file_path
        self.sequence_length = sequence_length
        self.frame_step_size = frame_step_size
        self.min_proportion_of_after_event_frames = min_proportion_of_after_event_frames
        self.max_proportion_of_after_event_frames = max_proportion_of_after_event_frames
        self.num_sequences_for_no_event_videos = num_sequences_for_no_event_videos
        self.format_frame_fn = format_frame_fn
    
    def __call__(self):
        labels_df = pd.read_csv(self.labels_file_path)
        
        videos_dir = pathlib.Path(self.videos_dir_path)
        
        for _, labels_row in labels_df.iterrows():
            video_filename = labels_row['video_filename']
            video_filepath = videos_dir / video_filename
            video_filepath = str(video_filepath.resolve())
            
            event_timestamp_millis = labels_row['event_timestamp_millis']
            
            frames_seqs = []
            labels = []
            
            if event_timestamp_millis >= 0:
                frames_seqs, labels = get_frames_surrounding_event(video_filepath=video_filepath,
                                                                   event_timestamp_millis=event_timestamp_millis,
                                                                   sequence_length=self.sequence_length,
                                                                   frame_step=self.frame_step_size,
                                                                   min_proportion_of_after_event_frames=self.min_proportion_of_after_event_frames,
                                                                   max_proportion_of_after_event_frames=self.max_proportion_of_after_event_frames,
                                                                   format_frame_fn=self.format_frame_fn)
            else:
                frames_seqs = get_random_video_frames(video_filepath,
                                                      seq_length=self.sequence_length,
                                                      step_size=self.frame_step_size,
                                                      num_seqs=self.num_sequences_for_no_event_videos,
                                                      format_frame_fn=self.format_frame_fn)
                
                labels = np.zeros(shape=(self.num_sequences_for_no_event_videos, self.sequence_length))
            
            for (frames, label) in zip(frames_seqs, labels):
                yield frames, label.any().astype(np.uint8)
                
# test
if __name__ == "__main__":
    labels_file_path = './labels.csv'
    videos_dir_path = './data/hand_collision_videos'

    frame_generator = FrameGenerator(videos_dir_path=videos_dir_path, labels_file_path=labels_file_path, sequence_length=15,
                                     frame_step_size=5, min_proportion_of_after_event_frames=0.3, max_proportion_of_after_event_frames=0.8,
                                     num_sequences_for_no_event_videos=5)

    for frames, label in frame_generator():
        assert frames.shape == (15, 720, 1280, 3)
        assert frames.dtype == np.uint8
        assert label.shape == ()
        assert label.dtype == np.uint8
        if DEBUG:
            print(frames.shape)
            print(frames.dtype)
            print(label.shape)
            print(label.dtype)
            print(label)

    print('All tests OK.')