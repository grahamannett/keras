'''
    example:
    dataset being [1,2,3,..,98,99,100]
    could be any k features by N time steps
    x_window = 4
    y_window = 2

    x_step = 2
        https://gist.github.com/braingineer/c69482eb1bfa4ac3bf9a7bc9b6b35cdf
        Generate minibatches on sequences with
    data augmentation.

    # Arguments
        step_size: set input mean to 0 over the dataset.
        y_length: number of elements to use in y

    example:
    step
    X                y
    [1,2,3,4],       [5,6]
    [3,4,5,6],       [7,8]
    [5,6,7,8],       [9,10]
    ...
    [95,96,97,98],[99,100]

also:
    example data can be generated with these:
        c1 = (list(range(1,10,1)) + list(range(10,1,-1)))*100
        c2 = [x * -1 * random() for x in c1]
        c1 = [x * random() for x in c1]
        d = list(zip(*[c1,c2]))
        d = np.array(d)


        sg = SequenceDataGen()
        t.flow(d)

EXAMPLE2 :
['preface\n\n\nsupposing that truth is a woma',
'face\n\n\nsupposing that truth is a woman--',
'e\n\n\nsupposing that truth is a woman--wha']
self.x_window = 40
x_window=10,
y_window=1,
x_step=3,
x_y_diff=0,
'''



########################################################
# lstm text generation example

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

path = get_file('nietzsche.txt',
                origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

x_win = 40
y_win = 1
chars_length = set(text)
# generator object with commented variables equivalent in lstm example
seqGen = SequenceDataGen(x_window=x_win,  # maxlen
                         x_step=3,  # step
                         y_window=1,  # next_char
                         x_y_diff=0,  # not in example but diff between sentences,next_char is 0
                         batch_size=16,  # batch_size
                         vocab=True)

seqGen.flow(text)

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(x_win, chars_length)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(chars_length))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
