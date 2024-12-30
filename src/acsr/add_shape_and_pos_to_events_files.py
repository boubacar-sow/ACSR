import os
import glob
import pandas as pd

from utils import get_word_code

manual = '_manual' # either '', or '_manual'
path2stimuli = '../stimuli/words/mp4'
fns_events = glob.glob(os.path.join(path2stimuli, f'*{manual}.events'))

# ADD SHAPE AND POSITION TO EVENTS FILE
for fn_events in fns_events:
    print(f'Processing file {fn_events}')
    word = open(fn_events, 'r').readlines()[0].strip('\n')
    df = pd.read_csv(fn_events, skiprows=1, index_col=False)
    
    poss, shapes = [], []
    for i_row, row in df.iterrows():
        if row['event'] == "SYLLABLE ONSET":
            syllable = row['stimulus'].strip()
            word_code = get_word_code(syllable)
            if word_code is not None:
                shape, pos = list(word_code)
            else:
                shape, pos = None, None
        else:
            shape, pos = None, None
        poss.append(pos)
        shapes.append(shape)
    
    df['pos'] = poss
    df['shape'] = shapes
    
    df.index.name = word
    
    fn_new_folder = os.path.dirname(fn_events)
    fn_new_base = os.path.basename(fn_events).split('.')
    
    fn_new_base = f'{fn_new_base[0]}.{fn_new_base[1]}_with_pos_shape.{fn_new_base[2]}'
    
    fn_new = os.path.join(fn_new_folder, fn_new_base)
    
    print(f'Saving new file to: {fn_new}')
    df.to_csv(fn_new)
    
