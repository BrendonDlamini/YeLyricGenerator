import csv

import numpy as np
import pandas as pd
import os
from keras.utils import np_utils
from textgenrnn import textgenrnn

# from google.colab import files

lyrics = []
i = 0

with open('data/kanye_verses.txt', 'r', encoding="utf-8") as filehandle:
    lines = filehandle.readlines()
    for line in lines:
        if not line.strip():
            continue
        else:
            lyrics.append(line)

lyrics_data = pd.DataFrame({'Lyrics': lyrics})

with open('data/lyricsText.txt', 'w', encoding="utf-8") as filehandle:
    for item in lyrics:
        filehandle.write(item + '\n')

textFileName = 'data/lyricsText.txt'
raw_text = open(textFileName, encoding="utf-8").read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
int_char = dict((i, c) for i, c in enumerate(chars))
char_int = dict((i, c) for c, i in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

print('Total Characters : ', n_chars)
print('Total Vocab : ', n_vocab)

model_name = '500nds_12Lrs_100epchs_Model'
textgen = textgenrnn(name=model_name)
textgen.train_from_file('data/kanye_verses.txt', num_epochs=1)


print(textgen.model.summary())

generated_characters = 300
textgen.generate_samples(300)
textgen.generate_to_file('date/lyrics.txt', 300)
