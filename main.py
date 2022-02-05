import numpy as np
import pandas as pd
import os
from keras.utils import np_utils
from textgenrnn import textgenrnn
from google.colab import files
dataset = pd.read_csv('data/taylor_swift_lyrics.csv', encoding="latin1")
dataset.head()

def processFirstLine(lyrics, songID, songName, row):
    lyrics.append(row['lyric'] + '\n')
    songID.append(row['year']*100 + row['track_n'])
    songName.append(row['track_title'])
    return lyrics, songID, songName

lyrics = []
songID = []
songName = []

songNumber = 1

i = 0
isFirstLine = True

for index, row in dataset.iterrows():
    if(songNumber == row['track_n']):
        if(isFirstLine):
            lyrics, songID, songName = processFirstLine(lyrics, songID, songName, row)
            isFirstLine = False
        else:
            lyrics[i] += row['lyric'] + '\n'
    else:
        lyrics, songID, songName = processFirstLine(lyrics, songID, songName, row)
        songNumber = row['track_n']
        i += 1

lyrics_data = pd.DataFrame({'songID':songID, 'songName':songName, 'lyrics':lyrics})

with open('data/lyricsText.txt', 'w', encoding="utf-8") as filehandle:
    for listitem in lyrics:
        filehandle.write('%s\n' % listitem)


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

seq_len = 100
data_x = []
data_y = []

for i in range(0, n_chars - seq_len, 1):
    seq_in = raw_text[i:i+seq_len]
    seq_out = raw_text[i + seq_len]
    data_x.append(char_int[char] for char in seq_in)
    data_y.append(char_int[seq_out])

n_patterns = len(data_x)
print('Total Patterns : ', n_patterns)

x = np.reshape(data_x, (n_patterns, seq_len, 1))
x = x/float(n_vocab)
y = np_utils.to_categorical(data_y)


model_cfg = {
    'rnn_size': 500,
    'rnn_layers': 12,
    'rnn_bidirectional': True,
    'max_length': 15,
    'max_words': 10000,
    'dim_embeddings': 100,
    'word_level': False,
}
train_cfg = {
    'line_delimited': True,
    'num_epochs': 100,
    'gen_epochs': 25,
    'batch_size': 750,
    'train_size': 0.8,
    'dropout': 0.0,
    'max_gen_length': 300,
    'validation': True,
    'is_csv': False
}

# uploaded = files.upload()
all_files = [(name, os.path.getmtime(name)) for name in os.listdir()]
latest_file = sorted(all_files, key=lambda x: -x[1])[0][0]

model_name = '500nds_12Lrs_100epchs_Model'
textgen = textgenrnn(name=model_name)
train_function = textgen.train_from_file if train_cfg['line_delimited'] else textgen.train_from_largetext_file
train_function(
    file_path=latest_file,
    new_model=True,
    num_epochs=train_cfg['num_epochs'],
    gen_epochs=train_cfg['gen_epochs'],
    batch_size=train_cfg['batch_size'],
    train_size=train_cfg['train_size'],
    dropout=train_cfg['dropout'],
    max_gen_length=train_cfg['max_gen_length'],
    validation=train_cfg['validation'],
    is_csv=train_cfg['is_csv'],
    rnn_layers=model_cfg['rnn_layers'],
    rnn_size=model_cfg['rnn_size'],
    rnn_bidirectional=model_cfg['rnn_bidirectional'],
    max_length=model_cfg['max_length'],
    dim_embeddings=model_cfg['dim_embeddings'],
    word_level=model_cfg['word_level'])

print(textgen.model.summary())


# files.download('{}_weights.hdf5'.format(model_name))
# files.download('{}_vocab.json'.format(model_name))
# files.download('{}_config.json'.format(model_name))

textgen = textgenrnn(weights_path='6layers30EpochsModel_weights.hdf5',
                       vocab_path='6layers30EpochsModel_vocab.json',
                       config_path='6layers30EpochsModel_config.json')

generated_characters = 300
textgen.generate_samples(300)
textgen.generate_to_file('date/lyrics.txt', 300)


