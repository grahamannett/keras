<<<<<<< Updated upstream
'''
    example:
    dataset being [1,2,3,..,98,99,100]
    could be any k features by N time steps
    x_window = 4
    y_window = 2

    x_step = 2

=======
class SequenceDataGen(object):
        '''
        https://gist.github.com/braingineer/c69482eb1bfa4ac3bf9a7bc9b6b35cdf
        Generate minibatches on sequences with
    data augmentation.

    # Arguments
        step_size: set input mean to 0 over the dataset.
        y_length: number of elements to use in y

    example:
    step
>>>>>>> Stashed changes
    X                y
    [1,2,3,4],       [5,6]
    [3,4,5,6],       [7,8]
    [5,6,7,8],       [9,10]
    ...
<<<<<<< Updated upstream
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

    '''
import numpy as np

class SequenceDataGen:
    '''Generate batches on time series/sequential data
    # Arguments
        step_size: set input mean to 0 over the dataset.

        x_window: size of moving window over x
        y_window: size of moving window over y

        x_step: step between each x dataset
        y_step: step between each y dataset, should be identical to x_step

        x_y_diff: difference between x,y windows.
                    if 0, implies y window starts where x window stopped
                    if >=1, y window starts that many steps following x window
                    if <= -1, then the x_window, y_window will overlap

        batch_size: number of x,y training sequences to return
        shuffle: whether sequences should be shuffled or take t_1, ..., t_n
        vocab: if it is a text sequence example, creates hashing and stuff

    '''

    def __init__(self,
                 x_window=10,
                 y_window=1,
                 x_step=1,
                 x_y_diff=0,
                 batch_size=32,
                 shuffle=False,
                 vocab=False
                 ):

        self.x_window = x_window
        self.y_window = y_window

        self.x_step = x_step
        self.y_step = x_step

        self.x_y_diff = x_y_diff

        self.batch_index = 0
        self.total_batches_seen = 0

        self.batch_size = batch_size
        self.shuffle = shuffle
=======
    [99,100,101,102],[103,104]
    '''

    def __init__(self,
                 step_size=1,
                 y_length=1):
        self.x = x
        self.y = y
        self.x_step = 1
        self.x_size = x.shape
        self.y_step = 1
        self.y_size = y.shape
        self.x_y_size = 1
>>>>>>> Stashed changes

    def reset(self):
        self.batch_index = 0

    def move_window(self):
        pass

<<<<<<< Updated upstream
    def gen_possible_indices(self, X):
        '''
        get possible indices's for time series given
        '''
        if isinstance(X, str):
            X_shape = len(X)
        else:
            X_shape = X.shape[0]
        last_window = self.x_window + self.y_window + self.x_y_diff
        idxes = list(range(0, X_shape - last_window, self.x_step))
        if self.shuffle:
            from random import shuffle
            shuffle(idxes)
        return idxes

    def generate_x_y(self, data, start_index):
        X_window_end = start_index + self.x_window
        X_window = data[start_index: X_window_end]
        y_start = X_window_end + self.x_y_diff
        y_end = y_start + self.y_window
        Y_window = data[y_start:y_end]
        return X_window, Y_window

    def _flow_single(self, data, seed=None):
        for idx in self.idxes:
            yield self.generate_x_y(data, idx)

    def _flow_batch(self, data, seed=None):
        idxes_ = self.idxes_
        while idxes_:
            batch_ = idxes_[:self.batch_size]
            del idxes_[:self.batch_size]
            gend = [self.generate_x_y(data, b) for b in batch_]
            yield gend

    def flow(self, data, seed=None):
        self.idxes = self.gen_possible_indices(data)
        # self.flow_generator = self._flow_single(data)
        self.flow_generator = self._flow_batch(data)
        return self

    def flow_from_vocab(self, data, words=False):
        '''vocab from:
        https://gist.github.com/braingineer/c69482eb1bfa4ac3bf9a7bc9b6b35cdf
        github.com/braingineer/ikelos/blob/master/ikelos/data/data_server.py

        other notes:
        github.com/fchollet/keras/blob/master/examples/lstm_text_generation.

        '''
        # split by words or chars
        if words:
            # # implement later
            # things_to_replace = {'\n': ' ', '--': ' -- '}
            # for k, v in things_to_replace.items():
            #     data = data.replace(k, v)
            # self.tokens = set(self.data.split(' '))
            pass
        else:
            self.tokens = set(data)

        self.idxes = self.gen_possible_indices(data)

        self.token_indices = dict((c, i) for i, c in enumerate(self.tokens))
        self.indices_token = dict((i, c) for i, c in enumerate(self.tokens))
        # vectorization

        X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

    def __iter__(self):
=======
    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        while 1:
            index_array = np.arange(N)
            if self.batch_index == 0:
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def flow(self, X, y, batch_size=32, shuffle=False, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.reset()
        self.flow_generator = self._flow_index(X.shape[0], batch_size,
                                               shuffle, seed)
        return self

            def __iter__(self):
>>>>>>> Stashed changes
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

<<<<<<< Updated upstream
'''       EXAMPLE :
        ['preface\n\n\nsupposing that truth is a woma',
        'face\n\n\nsupposing that truth is a woman--',
        'e\n\n\nsupposing that truth is a woman--wha']
        self.x_window = 40
        x_window=10,
        y_window=1,
        x_step=3,
        x_y_diff=0,'''
=======
    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.flow_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        bX = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            x = self.random_transform(x.astype('float32'))
            x = self.standardize(x)
            bX[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(bX[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}.{format}'.format(prefix=self.save_prefix,
                                                           index=current_index + i,
                                                           format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        bY = self.y[index_array]
        return bX, bY

    def __next__(self):
        # for python 3.x.
        return self.next()

>>>>>>> Stashed changes
