import numpy as np
import collections
import os
import tensorflow as tf
from itertools import groupby

all_words = []

def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """
    
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths

def _read_words(filename):
    #with tf.gfile.GFile(filename, "r") as f:
    #    w = f.read().decode("utf-8").replace("\n", " <eos> ").split(" ")
    unknown = "XXXXXXXXX"
    w = np.load(filename)
    pos = np.where(w==unknown)
    for p in pos:
        w = np.insert(w, p+1, "<eos>")
    return w

def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    global all_words
    all_words = words
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]
    #+[word_to_id["<eos>"]]


def load_y_labels(data_path="data/"):

    train_path = os.path.join(data_path, "train_y.txt")
    valid_path = os.path.join(data_path, "val_y.txt")
    test_path = os.path.join(data_path, "test_y.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    train_data_batch_ = [list(group) for k, group in groupby(train_data, lambda x: x == word_to_id['<eos>']) if not k]
    valid_data_batch_ = [list(group) for k, group in groupby(train_data, lambda x: x == word_to_id['<eos>']) if not k]
    test_data_batch_ = [list(group) for k, group in groupby(train_data, lambda x: x == word_to_id['<eos>']) if not k] 
    return train_data_batch_, valid_data_batch_, test_data_batch_, vocabulary

def pre_process_embeddings(data_path="data/"):
    trainy_path = os.path.join(data_path, 'train_y.npy')    
    word_to_id = _build_vocab(trainy_path)
    embed_path = os.path.join(data_path, "embeddings.npy")
    embed = np.load(embed_path)
    embed = embed[()]
    embedding_narr = np.array([ [0]*62 ]*len(embed.keys()))
    # np.zeros(len(embed.keys()))
    for key, value in embed.iteritems():
        if key in word_to_id:
            if len(value) == 62:
                embedding_narr[word_to_id[key]] = value
        
    print("Saving embeddings array..")
    np.save("data/embeddings_done.npy",embedding_narr)
    print("Saved embeddings array..")

def m_load_data(data_path="data/"):

    trainx_path = os.path.join(data_path, "train_x.npy")
    trainy_path = os.path.join(data_path, 'train_y.npy')
    
    validx_path = os.path.join(data_path, "val_x.npy")
    validy_path = os.path.join(data_path, "val_y.npy")
    
    testx_path = os.path.join(data_path, "test_x.npy")
    testy_path = os.path.join(data_path, "test_y.npy")
    
    # Train data prep
    trainx = np.load(trainx_path)
    word_to_id = _build_vocab(trainy_path)
    trainy = _file_to_word_ids(trainy_path, word_to_id)
    vocabulary = len(word_to_id)
    
    trainx_data_batch_ = []
    last_index = 0
    curr_index = 0
    
    while True:
        try:
            curr_index = trainy.index(word_to_id['<eos>'], last_index)
            temp = trainx[0][last_index:curr_index]
            trainx_data_batch_.append(temp)
            last_index = curr_index + 1
        except ValueError as _:
            break
    
    trainy_data_batch_ = [list(group) for k, group in groupby(trainy, lambda x: x == word_to_id['<eos>']) if not k]
    
    # Test data prep
    
    testx = np.load(testx_path)
    testy = _file_to_word_ids(testy_path, word_to_id)
    
    testx_data_batch_ = []
    last_index = 0
    curr_index = 0
    
    while True:
        try:
            curr_index = testy.index(word_to_id['<eos>'], last_index)
            temp = testx[0][last_index:curr_index]
            testx_data_batch_.append(temp)
            last_index = curr_index + 1
        except ValueError as _:
            break
    
    testy_data_batch_ = [list(group) for k, group in groupby(testy, lambda x: x == word_to_id['<eos>']) if not k]
    
    # Validation data prep
    
    valx = np.load(validx_path)
    valy = _file_to_word_ids(validy_path, word_to_id)
    
    valx_data_batch_ = []
    last_index = 0
    curr_index = 0
    
    while True:
        try:
            curr_index = valy.index(word_to_id['<eos>'], last_index)
            temp = valx[0][last_index:curr_index]
            valx_data_batch_.append(temp)
            last_index = curr_index + 1
        except ValueError as _:
            break
    
    valy_data_batch_ = [list(group) for k, group in groupby(valy, lambda x: x == word_to_id['<eos>']) if not k]
    
    return trainx_data_batch_, trainy_data_batch_, valx_data_batch_, valy_data_batch_, testx_data_batch_, testy_data_batch_, vocabulary
    
def read_and_load_data(batch, batch_size):
    l = len(batch)
    for ndx in range(0, l, batch_size):
        yield batch[ndx:min(ndx + batch_size, l)]    
    
def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]