# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:37:05 2022

"""
import csv
import logging
import os
import pickle
import sys

import cv2  # Import opencv
import matplotlib.pyplot as plt
import mediapipe as mp  # Import mediapipe
import numpy as np
import pandas as pd
import textgrids
from scipy.signal import argrelextrema, savgol_filter
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm


def load_model(filename):
    with open(filename, 'rb') as f:
        model, feature_names = pickle.load(f)
    return model, feature_names


def load_video(path2file):
    cap = cv2.VideoCapture(path2file)
    cap.set(3,640) # camera width
    cap.set(4,480) # camera height
    return cap


def extract_class_from_fn(fn):
    '''
    get class number from filename, e.g.,
    '4' from 'position_04.mp4'
    '''
    if fn is not None:
        st = fn.find('_') + 1
        ed = fn.find('.')
        c = fn[st:ed]#.split('_')[0]
        return int(c)
    else:
        return None


def get_distance(df_name, landmark1, landmark2, norm_factor=None):
    '''


    Parameters
    ----------
    df_name : TYPE
        DESCRIPTION.
    landmark1 : STR
        name of first landmark (e.g., hand20)
    landmark2 : STR
        name of second landmark (e.g., face234)

    Returns
    -------
    series for dataframe
    The distance between landmark1 and landmark2

    '''

    x1 = df_name[f'x_{landmark1}']
    x2 = df_name[f'x_{landmark2}']
    y1 = df_name[f'y_{landmark1}']
    y2 = df_name[f'y_{landmark2}']
    z1 = df_name[f'z_{landmark1}']
    z2 = df_name[f'z_{landmark2}']
    d = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

    # NORMALIZE
    if norm_factor is not None:
        d /= norm_factor

    return  d

def get_delta_dim(df_name, landmark1, landmark2, dim, norm_factor=None):
    delta = df_name[f'{dim}_{landmark1}'] - df_name[f'{dim}_{landmark2}']
    # NORMALIZE
    if norm_factor is not None:
        delta /= norm_factor
    return  delta


def get_frames_around_event(fn_video, frame_number, n_neighbor_frames):
    st = frame_number - n_neighbor_frames
    ed = frame_number + n_neighbor_frames + 1
    frame_numbers = range(st, ed)

    extracted_frames = []
    cap = cv2.VideoCapture(fn_video)

    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            extracted_frames.append(frame)
    cap.release()
        
    return extracted_frames


def create_video_from_frames(fn_video, extracted_frames):
    out = None
    if extracted_frames:
        height, width, _ = extracted_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 video
        out = cv2.VideoWriter(fn_video, fourcc, 30.0, (width, height))
        for frame in extracted_frames:
            out.write(frame)
        out.release()
    
    return out


def extract_coordinates(cap, fn_video, show_video=False, verbose=True):
    """
    Extract hand, lip, and specific face landmarks from a video using MediaPipe Holistic.

    Args:
        cap: Video capture object.
        fn_video (str): Name of the video file.
        show_video (bool): Whether to display the video with landmarks.
        verbose (bool): Whether to print progress information.

    Returns:
        pd.DataFrame: DataFrame containing hand, lip, and specific face landmarks for each frame.
    """
    if verbose:
        print(f"Extracting coordinates for: {fn_video}")
    
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic

    # Define columns for the DataFrame
    columns = ["fn_video", "frame_number"]

    # Define real indices for lip, hand, and specific face landmarks
    lip_indices = [17, 314, 405, 321, 375, 291, 84, 181, 91, 146, 
                   0, 267, 269, 270, 409, 40, 37, 39, 40, 185, 
                   61, 78, 95, 88, 87, 14, 317, 402, 324, 308, 
                   80, 81, 82, 13, 312, 311, 319, 308]  # 38 lip landmarks
    hand_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # 21 hand landmarks
    face_indices = [454, 234, 200, 214, 50]  # Specific face landmarks

    # Add columns for hand landmarks (21 keypoints, x, y, z)
    for i in hand_indices:
        columns += [f"hand_x{i}", f"hand_y{i}", f"hand_z{i}"]

    # Add columns for lip landmarks (39 keypoints, x, y, z)
    for i in lip_indices:
        columns += [f"lip_x{i}", f"lip_y{i}", f"lip_z{i}"]

    # Add columns for specific face landmarks (x, y, z)
    for i in face_indices:
        columns += [f"face_x{i}", f"face_y{i}", f"face_z{i}"]

    # Initialize DataFrame
    df_coords = pd.DataFrame(columns=columns)

    # Get video properties
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print(f"Number of frames in video: {n_frames}")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Frames per second: {fps}")
        video_length = n_frames / fps
        print(f"Video length: {video_length} seconds")
    pbar = tqdm(total=n_frames)

    # Initiate holistic model
    i_frame = 0
    with mp_holistic.Holistic(
        min_detection_confidence=0.2, min_tracking_confidence=0.2,
        static_image_mode=False,  # Set to False for video processing
        model_complexity=2,  # Adjust model complexity (0=light, 1=medium, 2=heavy)
        enable_segmentation=False,  # Disable segmentation for faster processing
        smooth_landmarks=True,  # Smooth landmarks for better tracking
        refine_face_landmarks=True  # Refine face landmarks for better accuracy
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            i_frame += 1

            if not ret:
                break

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks on the frame (optional)
            if show_video:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                )
                cv2.imshow("cued_estimated", image)

            # Extract hand landmarks
            hand_landmarks = []
            if results.right_hand_landmarks:
                for i in hand_indices:
                    landmark = results.right_hand_landmarks.landmark[i]
                    hand_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                hand_landmarks.extend([None] * len(hand_indices)*3)  # 21 keypoints × 3 coordinates

            # Extract lip landmarks (subset of face landmarks)
            lip_landmarks = []
            if results.face_landmarks:
                for i in lip_indices:
                    landmark = results.face_landmarks.landmark[i]
                    lip_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                lip_landmarks.extend([None] * len(lip_indices)*3)  # 39 keypoints × 3 coordinates

            # Extract specific face landmarks (face_454 and face_234)
            face_specific_landmarks = []
            if results.face_landmarks:
                for i in face_indices:
                    landmark = results.face_landmarks.landmark[i]
                    face_specific_landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                face_specific_landmarks.extend([None] * len(face_indices)*3)  # 2 keypoints × 3 coordinates

            # Create the row for the DataFrame
            row = [fn_video, i_frame] + hand_landmarks + lip_landmarks + face_specific_landmarks
            curr_df = pd.DataFrame([row], columns=columns)
            df_coords = pd.concat([df_coords, curr_df], ignore_index=True)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()

    # Verify that all frames were processed
    assert n_frames - df_coords.shape[0] <= 1

    return df_coords

def extract_features(df_coords):
    """
    Extract features from the coordinates DataFrame with corrected tangent calculation
    and dual normalization (face width and hand span).
    """
    df_features = pd.DataFrame()
    df_features["fn_video"] = df_coords["fn_video"].copy()
    df_features["frame_number"] = df_coords["frame_number"]
    
    # Compute normalization factors
    face_width = get_distance(df_coords, "face_x454", "face_x234").mean()
    hand_span = get_distance(df_coords, "hand_x0", "hand_x12").mean()
    
    # Fallback to face width if hand span is invalid (e.g., no hand detected)
    if np.isnan(hand_span) or hand_span == 0:
        hand_span = face_width
    
    # HAND-FACE DISTANCES AND ANGLES (normalize by face width)
    position_index_pairs = get_index_pairs("position")
    for hand_index, face_index in position_index_pairs:
        # Distance
        feature_name = f"distance_face{face_index}_hand{hand_index}"
        df_features[feature_name] = get_distance(
            df_coords, 
            f"face_x{face_index}", 
            f"hand_x{hand_index}", 
            norm_factor=face_width  # Use face width for hand-face distances
        )
        
        # Angle calculation using arctan2(dy, dx) instead of dx/dy
        dx = get_delta_dim(df_coords, f"face_x{face_index}", f"hand_x{hand_index}", "x", norm_factor=face_width)
        dy = get_delta_dim(df_coords, f"face_x{face_index}", f"hand_x{hand_index}", "y", norm_factor=face_width)
        angle_radians = np.arctan2(dy, dx)  # Bounded between -π and π
        df_features[f"angle_face{face_index}_hand{hand_index}"] = angle_radians

    # HAND-HAND DISTANCES (normalize by hand span)
    shape_index_pairs = get_index_pairs("shape")
    for hand_index1, hand_index2 in shape_index_pairs:
        feature_name = f"distance_hand{hand_index1}_hand{hand_index2}"
        df_features[feature_name] = get_distance(
            df_coords, 
            f"hand_x{hand_index1}", 
            f"hand_x{hand_index2}", 
            norm_factor=hand_span  # Use hand span for hand-hand distances
        )
    
    # FINGER ANGLES (using normalized hand coordinates)
    df_features["thumb_index_angle"] = get_angle(
        df_coords["hand_x4"], df_coords["hand_y4"], df_coords["hand_z4"],
        df_coords["hand_x0"], df_coords["hand_y0"], df_coords["hand_z0"],
        df_coords["hand_x8"], df_coords["hand_y8"], df_coords["hand_z8"]
    ) / hand_span  # Normalize by hand span

    # LIP FEATURES (keep existing normalization by face width)
    df_features["lip_width"] = get_distance(df_coords, "lip_x61", "lip_x291", norm_factor=face_width)
    df_features["lip_height"] = get_distance(df_coords, "lip_x0", "lip_x17", norm_factor=face_width)
    df_features["lip_area"] = df_features["lip_width"] * df_features["lip_height"]
    
    # Lip curvature (normalize by face width)
    df_features["lip_curvature"] = get_angle(
        df_coords["lip_x61"], df_coords["lip_y61"], df_coords["lip_z61"],
        df_coords["lip_x0"], df_coords["lip_y0"], df_coords["lip_z0"],
        df_coords["lip_x17"], df_coords["lip_y17"], df_coords["lip_z17"]
    ) / face_width
    
    # MOTION FEATURES (normalize velocities by face width)
    df_features["lip_velocity_x"] = df_coords["lip_x0"].diff() / face_width
    df_features["lip_velocity_y"] = df_coords["lip_y0"].diff() / face_width
    df_features["lip_acceleration_x"] = df_features["lip_velocity_x"].diff()
    df_features["lip_acceleration_y"] = df_features["lip_velocity_y"].diff()
    
    # Hand motion features (normalize by hand span)
    df_features["hand8_velocity_x"] = df_coords["hand_x8"].diff() / hand_span
    df_features["hand8_velocity_y"] = df_coords["hand_y8"].diff() / hand_span
    
    return df_features


def get_index_pairs(property_type):
    """
    Get index pairs for hand-face or hand-hand distances.

    Args:
        property_type (str): 'position' for hand-face distances, 'shape' for hand-hand distances.

    Returns:
        list: List of index pairs.
    """
    index_pairs = []
    if property_type == 'shape':
        # Hand-hand distances for hand shapes
        index_pairs.extend([
            (2, 4), (5, 8), (9, 12), (13, 16), (17, 20),  # Finger tips to mcp
            (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),    # Wrist to finger tips
            (4, 5), (4, 8), (8, 12), (7, 11), (6, 10), (12, 16), (16, 20)    # Adjacent finger tips
        ])
    elif property_type == 'position':
        # Hand-face distances for hand position
        hand_indices = [8, 9, 12]  # Index and middle fingers
        face_indices = [234, 454, 200, 214, 50]  # Specific face landmarks
        for hand_index in hand_indices:
            for face_index in face_indices:
                index_pairs.append((hand_index, face_index))
    return index_pairs


def get_feature_names(property_name):
    """
    Get feature names for hand-face or hand-hand distances.

    Args:
        property_name (str): 'position' for hand-face distances, 'shape' for hand-hand distances.

    Returns:
        list: List of feature names.
    """
    feature_names = []
    if property_name == 'position':
        position_index_pairs = get_index_pairs('position')
        for hand_index, face_index in position_index_pairs:
            feature_name = f'distance_face{face_index}_hand{hand_index}'
            feature_names.append(feature_name)
            feature_name = f'tan_angle_face{face_index}_hand{hand_index}'
            feature_names.append(feature_name)
    elif property_name == 'shape':
        shape_index_pairs = get_index_pairs('shape')
        for hand_index1, hand_index2 in shape_index_pairs:
            feature_name = f'distance_hand{hand_index1}_hand{hand_index2}'
            feature_names.append(feature_name)
    return feature_names


def get_distance(df_name, landmark1, landmark2, norm_factor=None):
    """
    Compute the Euclidean distance between two landmarks.

    Args:
        df_name (pd.DataFrame): DataFrame containing landmark coordinates.
        landmark1 (str): Name of the first landmark (e.g., "hand_x8").
        landmark2 (str): Name of the second landmark (e.g., "face_x234").
        norm_factor (float, optional): Normalization factor. Defaults to None.

    Returns:
        pd.Series: Distance between the two landmarks.
    """
    x1 = df_name[f"{landmark1}"]
    x2 = df_name[f"{landmark2}"]
    y1 = df_name[f"{landmark1.replace('_x', '_y')}"]
    y2 = df_name[f"{landmark2.replace('_x', '_y')}"]
    z1 = df_name[f"{landmark1.replace('_x', '_z')}"]
    z2 = df_name[f"{landmark2.replace('_x', '_z')}"]
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    # Normalize
    if norm_factor is not None:
        d /= norm_factor

    return d


def get_delta_dim(df_name, landmark1, landmark2, dim, norm_factor=None):
    """
    Compute the difference in a specific dimension (x, y, or z) between two landmarks.

    Args:
        df_name (pd.DataFrame): DataFrame containing landmark coordinates.
        landmark1 (str): Name of the first landmark (e.g., "hand_x8").
        landmark2 (str): Name of the second landmark (e.g., "face_x234").
        dim (str): Dimension to compute the difference for ("x", "y", or "z").
        norm_factor (float, optional): Normalization factor. Defaults to None.

    Returns:
        pd.Series: Difference in the specified dimension.
    """
    delta = df_name[f"{landmark1.replace('_x', f'_{dim}')}"] - df_name[f"{landmark2.replace('_x', f'_{dim}')}"]
    # Normalize
    if norm_factor is not None:
        delta /= norm_factor
    return delta


def get_angle(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    """
    Compute the angle between three points.

    Args:
        x1, y1, z1: Coordinates of the first point.
        x2, y2, z2: Coordinates of the second point (vertex).
        x3, y3, z3: Coordinates of the third point.

    Returns:
        pd.Series: Angle in radians.
    """
    v1 = np.array([x1 - x2, y1 - y2, z1 - z2])
    v2 = np.array([x3 - x2, y3 - y2, z3 - z2])
    dot_product = np.sum(v1 * v2, axis=0)
    norm_v1 = np.linalg.norm(v1, axis=0)
    norm_v2 = np.linalg.norm(v2, axis=0)
    angle = np.arccos(dot_product / (norm_v1 * norm_v2))
    return angle

def setup_logging(loglevel):
    """Setup basic logging

    Args:
        loglevel (int): Minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def compute_predictions(model, df_features):
    '''
    model - sklean model
    df_features - dataframe with n_samples X n_features
    '''
    X = df_features.to_numpy()

    predicted_class, predicted_probs = [], []
    for X_i in X:
        if (None in X_i) or (np.nan in X_i) or any([xi!=xi for xi in X_i]):
            predicted_c = None
            predicted_p = None
        else:
            predicted_c = model.predict([X_i])[0]
            predicted_p = model.predict_proba([X_i])[0]
        predicted_class.append(predicted_c)
        predicted_probs.append(predicted_p)

    return np.asarray(predicted_probs, dtype=object), \
        np.asarray(predicted_class)


def compute_velocity(df, landmark, fn=None):
    frame_number = df['frame_number']
    x = df['x_' + landmark].values
    y = df['y_' + landmark].values
    z = df['z_' + landmark].values

    dx = np.gradient(x, frame_number)
    dy = np.gradient(y, frame_number)
    dz = np.gradient(z, frame_number)

    dx2 = np.gradient(dx, frame_number)
    dy2 = np.gradient(dy, frame_number)
    dz2 = np.gradient(dz, frame_number)

    v = np.sqrt(dx**2 + dy**2 + dz**2)
    a = np.sqrt(dx2**2 + dy2**2 + dz2**2)

    v_smoothed = savgol_filter(v, 9, 3) # window
    a_smoothed = savgol_filter(a, 9, 3) # window

    if fn is not None:
        fig, ax = plt.subplots()
        ax.plot(v_smoothed, lw=3, color='k')
        ax.plot(a_smoothed, lw=3, color='b')
        ax.set_xlabel('Frame', fontsize=16)
        ax.set_ylabel('Velocity', fontsize=16)
        ax.set_ylim([-0.01, 0.01])
        fig.savefig(fn + '.png')
    return  v_smoothed, a_smoothed


def get_phone_onsets(fn_textgrid):
    times, labels = [], []

    grid = textgrids.TextGrid(fn_textgrid)
    phones = grid['phones']
    for phone in phones:
        if phone.text.transcode() != '':
            times.append(phone.xmin)
            labels.append(phone.text.transcode())

    return times, labels


def get_stimulus_string(fn_video):
    fn_base = os.path.basename(fn_video)[:-4]
    fn_stimulus = fn_base + '.txt'
    fn_stimulus = os.path.join('ACSR/stimuli/words/mfa_in', fn_stimulus)
    s = open(fn_stimulus, 'r').readlines()
    return s[0].strip('\n')


def dict_phone_transcription():
    # Megalex (key) to MFA (value) phone labels
    d = {}
    d['R'] = 'ʁ'
    d['N'] = 'ɲ'
    d['§'] = 'ɔ̃'
    d['Z'] = 'ʒ'
    d['5'] = 'ɛ̃'
    d['E'] = 'ɛ'
    d['9'] = 'œ'
    d['8'] = 'ɥ'
    d['S'] = 'ʃ'
    d['O'] = 'ɔ'
    d['2'] = 'ø'
    d['g'] = 'ɟ'
    d['g'] = 'ɡ'
    d['@'] = 'ɑ̃'
    d['8'] = 'ɥ'
    return d

def find_syllable_onsets(lpc_syllables, times_phones, labels_phones):
    phones = labels_phones.copy()
    d_phone_transcription = dict_phone_transcription()
    #print(lpc_syllables)
    #[print(p, t) for p, t in zip(phones, times_phones)]
    #print('-'*100)
    times = []
    for syllable in lpc_syllables:
        first_phone = syllable[0]
        if first_phone in d_phone_transcription.keys():
            first_phone = d_phone_transcription[first_phone]
        for i, phone in enumerate(phones):
            if first_phone == phone:
                times.append(times_phones[i])
                del phones[i]
                del times_phones[i]
                break
    return times


def get_syllable_onset_frames_from_lpc_file(fn_video):
    fn_base = os.path.basename(fn_video)[:-4]

    # Get LPC parsing of stimulus, into separate SYLLABLES
    # (MFA is for ALL phones and we need to know which phones are at the beginning of each syllable)
    fn_lpc_parsing = fn_base + '.lpc'
    fn_lpc_parsing = os.path.join('ACSR/stimuli/words/txt', fn_lpc_parsing)
    lpc_syllables = open(fn_lpc_parsing, 'r').readlines()[0].strip('\n').split()

    return lpc_syllables

    return 
def get_syllable_onset_frames_from_mfa(fn_video, lpc_syllables):

    # Load video and get number of frames per second (fps)
    cap = load_video(fn_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # frames per second
    assert fps > 0; 'Frames per seconds is not a positive number'

    # Load corresponing TextGrid file
    fn_base = os.path.basename(fn_video)[:-4]
    fn_textgrid = fn_base + '.TextGrid'
    fn_textgrid = os.path.join('../stimuli/words/mfa_out', fn_textgrid)

    # Get LPC parsing of stimulus, into separate SYLLABLES
    # (MFA is for ALL phones and we need to know which phones are at the beginning of each syllable)
    #fn_lpc_parsing = fn_base + '.lpc'
    #fn_lpc_parsing = os.path.join('../stimuli/words/txt', fn_lpc_parsing)
    #lpc_syllables = open(fn_lpc_parsing, 'r').readlines()[0].strip('\n').split()

    # PHONE onests in seconds from MFA
    onset_secs_phones_mfa, labels_phones_textgrid = get_phone_onsets(fn_textgrid)
    print(onset_secs_phones_mfa, labels_phones_textgrid)
    # SYLLABLE ONSET from MFA based on the onset of their FIRST PHONE
    onset_secs_syllables_mfa = find_syllable_onsets(lpc_syllables, # in seconds
                                                    onset_secs_phones_mfa,
                                                    labels_phones_textgrid)
    onset_frames_syllables_mfa = [int(t*fps) for t in onset_secs_syllables_mfa] # in frames

    return onset_frames_syllables_mfa



def find_onsets_based_on_extrema(time_series,
                                 n_syllables=None,
                                 onset_frames_syllables_mfa=None,
                                 thresh=None): # condition: time_series > thresh

    if onset_frames_syllables_mfa is not None: 
        onset_frames_syllables_mfa = np.asarray(onset_frames_syllables_mfa)

    # find extrema
    onset_frames_extrema = argrelextrema(time_series, np.greater)[0]
    # Threshold
    if thresh is not None:
        onset_frames_extrema = np.asarray([onset_frame for onset_frame in onset_frames_extrema if time_series[onset_frame]>thresh])

    onset_frames_extrema_temp = onset_frames_extrema.copy()
    onset_frames_picked = []
    if onset_frames_syllables_mfa is not None: # use MFA onsets to constrain the solution
        if len(onset_frames_syllables_mfa) == len(onset_frames_extrema_temp):
            onset_frames_picked = onset_frames_extrema_temp
        else:
            for i_frame, onset_frame_syl_mfa in enumerate(onset_frames_syllables_mfa):
                # Find extremum that is nearest to current MFA onset
                delta = np.abs(onset_frames_extrema_temp - onset_frame_syl_mfa)
                IX_onset_frame_extremum_nearest_mfa = np.argmin(delta)
                onset_frame_extremum_nearest_mfa = onset_frames_extrema_temp[IX_onset_frame_extremum_nearest_mfa]
                onset_frames_picked.append(onset_frame_extremum_nearest_mfa)
                # Remove past indexes, in order to make sure the next onset frame is in the future
                onset_frames_extrema_temp = onset_frames_extrema_temp[onset_frames_extrema_temp > onset_frame_extremum_nearest_mfa]
                if len(onset_frames_extrema_temp)==0:
                    while len(onset_frames_picked) < len(onset_frames_syllables_mfa): # Fill None values if not enough identified extrema
                        onset_frames_picked.append(None)
                    break
    else:
        IXs = np.argpartition(onset_frames_extrema, -n_syllables)[-n_syllables:]
        onset_frames_picked = list(onset_frames_extrema[IXs])

    return onset_frames_picked, onset_frames_extrema

def scale_velocity(velocity):
    q25, q75 = np.percentile(velocity, 25), np.percentile(velocity, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    velocity = np.clip(velocity, lower, upper)
    velocity_scaled = minmax_scale(velocity)
    return velocity_scaled


def get_joint_measure(df_predictions_pos,
                      df_predictions_shape,
                      velocity_scaled,
                      weight_velocity=1):

    # MAX PROBABILITIES (POSITION AND SHAPE)
    max_probs_pos = df_predictions_pos.copy().filter(regex=("p_class*")).to_numpy().max(axis=1)
    max_probs_shape = df_predictions_shape.copy().filter(regex=("p_class*")).to_numpy().max(axis=1)
    probs_product = max_probs_pos * max_probs_shape
    # JOINT
    joint_measure = (weight_velocity * (1-velocity_scaled) + probs_product)/(1+weight_velocity)
    joint_measure_smoothed = savgol_filter(joint_measure, 15, 3) # window, smooth
    # replace nans caused by smoothing with original values
    is_nan_smoothed = np.isnan(joint_measure_smoothed)
    joint_measure_smoothed[is_nan_smoothed] = joint_measure[is_nan_smoothed]

    return joint_measure_smoothed


def write_onsets_to_file(str_stimulus, lpc_syllables, onset_frames_picked, fn_txt):
    
    # HACK TO EQUALIZE THE NUMBER OF EXPECTED ONSETS (NUM SYLLABLES) AND THE ONE FOUND
    if len(lpc_syllables) < len(onset_frames_picked): # REMOVE EXTRA ONSETS
        onset_frames_picked = onset_frames_picked[:3]
    for i_sy in range(len(lpc_syllables)-len(onset_frames_picked)): # ADD DUMMY ONSETS
        onset_frames_picked = list(onset_frames_picked)
        last_onset = onset_frames_picked[-1]
        onset_frames_picked.append(last_onset + i_sy + 1)

    assert len(lpc_syllables) == len(onset_frames_picked)

    with open(fn_txt, 'w') as f:
        f.write(f'{str_stimulus}\n')
        f.write('event,stimulus,frame_number\n')
        for (syllable, onset) in zip(lpc_syllables, onset_frames_picked):
            f.write(f'SYLLABLE ONSET, {syllable}, {onset}\n')
    return None

# FROM HAGAR

def get_LPC_p(word):
    lex = pd.read_csv("/home/yair/projects/ACSR/data/hagar/Lexique380.utf8.csv")
    lex = lex[(lex.ortho.str.contains('-| ') == False) & (lex.phon.str.contains('°') == False)]  # suppress schwa
    lex = lex.drop_duplicates(subset='ortho', keep="first")
    lex = lex[['ortho','phon', 'p_cvcv','nbhomogr','cv-cv','syll']]
    dic = lex.set_index('ortho').to_dict()

    cv_dic = dic['cv-cv']
    p_cv_dic = dic['syll']
    phon_dic = dic['phon']    

    dev_syl = pd.read_csv("/home/yair/projects/ACSR/data/hagar/lpc_syl_configurations.csv")
    dev_syl['lpc_n'] = dev_syl['LPC_config'].apply(lambda x: x.split('-'))
    dev_syl['lpc_n'] = dev_syl['lpc_n'].apply(lambda x: len(x))
    dic2 = dev_syl.set_index('spoken_config').to_dict()
    
    g_cv_dic = dic2['LPC_config']
    
    lpc_cv = get_LPC_cv(word, cv_dic, g_cv_dic)
    
    new_word = ''
    phon = phon_dic[word]
    if lpc_cv == cv_dic[word]:
        return p_cv_dic[word]
    else:
        l_lpc = lpc_cv.split('-')
        for syl in l_lpc:
            new_word += phon[:len(syl)]+'-'
            phon = phon[len(syl):]
        return new_word[:-1]


def get_LPC_cv(word, cv_dic, g_cv_dic):
    

    LPC_cv = ''
    if word in cv_dic:
        cv_lst = cv_dic[word].split('-')
        for syl in cv_lst:
            LPC_cv = LPC_cv + g_cv_dic[syl] + '-'
        return LPC_cv[:-1]

    else:
        return word

def get_word_code(syll):
    position = {'a': '0', 'o': '0', '9': '0', '5': '1', '2': '1', 'i': '2', '§': '2', '@': '2', 'E': '3', 'u': '3', 'O': '3', '1': '4', 'y': '4', 'e': '4'}
    configuration = {'p': '0', 'd': '0', 'Z': '0', 'k': '1', 'v': '1', 'z': '1', 's': '2', 'R': '2', 'b': '3', 'n': '3', '8': '3', 't': '4', 'm': '4', 'f': '4', 'l': '5', 'S': '5', 'N': '5', 'w': '5', 'g': '6', 'j': '7', 'G': '7'}
    try:
        code_word = ''
        if len(syll) == 1:
            if syll in configuration:
                code_word += configuration[syll]
                code_word += '0'
            else:
                code_word += '4'
                code_word += position[syll]
        else:
            for i in range (0,len(syll)):
                if syll[i] in configuration:
                    code_word += configuration[syll[i]]
                else:
                    code_word += position[syll[i]]
        return code_word
    except:
        return None


def shape_position_code(word):
    code_word = ""
    syll_lst = get_LPC_p(word).split("-")
    for syll in syll_lst:
        code_word += get_word_code(syll) + '-'  
    return code_word[:-1]
