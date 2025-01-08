import os
import glob
import pandas as pd

from utils import get_frames_around_event
from utils import create_video_from_frames

from tqdm import tqdm

debug = None

path2stimuli = '../stimuli/words/mp4'
path2out_video = '../data/training_videos/female/cropped/'

fns_events = glob.glob(os.path.join(path2stimuli, '*manual_with_pos_shape.events'))
print(f'Found {len(fns_events)} event files')
if len(fns_events)==0:
    raise('No events files found')

n_neighbor_frames = 2

# INIT FRAME COLLECTOR
frames_pos, frames_shape = {}, {}
for pos in range(5):
    frames_pos[pos] = []
for shape in range(8):
    frames_shape[shape] = []


for fn_events in tqdm(fns_events[:debug]):
    fn_video = fn_events[:-29]
    #print(f'Loading video: {fn_video}')
    
    # READ EVENT FILES
    word = open(fn_events, 'r').readlines()[0].strip('\n')
    df = pd.read_csv(fn_events)
    # LOOP OVER SYLLABLES
    for i_row, row in df.iterrows():
        if row['event'] == "SYLLABLE ONSET":
            # GET N FRAMES AROUND EVENT ONSET (N = 2 * num_neighbor_frames)
            shape, pos = row['shape'], row['pos']
            frame_number = row['frame_number']
            
            if any(pd.isnull([shape, pos])) or not isinstance(frame_number,
                                                              int):
                continue
            else:
                #print(f'Processing syllable {row["stimulus"]}, pos {pos}, shape {shape}')
                
                if (shape is not None) and (pos is not None):
                    #print(fn_video, frame_number, n_neighbor_frames)
                    frames = get_frames_around_event(fn_video,
                                                     frame_number,
                                                     n_neighbor_frames)
                    frames_pos[int(pos)].extend(frames)
                    frames_shape[int(shape)].extend(frames)            
        else:
            continue

# SAVE MERGED VIDEOS
print('Saving videos')
for pos in range(5):
    fn_merged_video = os.path.join(path2out_video, f'position_{pos}_from_words.mp4')
    out = create_video_from_frames(fn_merged_video, frames_pos[pos])
    if out is not None:
        print(f'Merged video was saved to {fn_merged_video}')

for shape in range(8):
    fn_merged_video = os.path.join(path2out_video, f'shape_{shape}_from_words.mp4')
    out = create_video_from_frames(fn_merged_video, frames_shape[shape])
    if out is not None:
        print(f'Merged video was saved to {fn_merged_video}')   

