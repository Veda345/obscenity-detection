from __future__ import division
from __future__ import print_function

from gensim.models import Word2Vec
from keras.models import Model
from keras.models import model_from_json
from keras.layers.embeddings import Embedding
from keras.layers import concatenate
from keras.layers import Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import shuffle

import numpy as np
# import matplotlib.pyplot as plt
import json

from util import getData, create_char_vocab_set


N_FILTER = 128

dense_outputs_chars = 32
dense_outputs_words = 1024
filter_kernels = [10, 8, 4, 3, 3, 3]
cat_output = 2

BATCH_SIZE = 100
BATCH_CNT = 1000
TEST_BATCH_CNT = 100
N_EPOCH = 5

MAX_CHAR_LEN = 1000
MAX_WORD_LEN = 20

EVAL_SET_SIZE = -1

vocab, reverse_vocab, char_vocab_size, check = create_char_vocab_set()

negative_file_path = ""
negative_obf_file_path = ""
positive_file_path = ""
positive_obf_file_path = ""


def get_negative_data():
    return getData(negative_file_path)


def get_negative_obf_data():
    return getData(negative_obf_file_path)


def get_positive_no_hu_data():
    return getData(positive_file_path)


def get_positive_no_hu_obf_data():
    return getData(positive_obf_file_path)


def load_w2v_vocab_and_weights(w2v_vocab_path):
    with open(w2v_vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data

    return word2idx


def read_file(get_data, obscene=False):
    X = get_data()
    if obscene:
        Y = to_categorical(np.ones(len(X)), 2)
    else:
        Y = to_categorical(np.zeros(len(X)), 2)
    return X, Y


def make_model(embedding_matrix):
    char_inputs = Input(shape=(MAX_CHAR_LEN, char_vocab_size), name='input', dtype='float32')

    conv = Conv1D(filters=N_FILTER, kernel_size=filter_kernels[0],
                  padding='valid', activation='relu',
                  kernel_initializer='glorot_normal', bias_initializer='zeros',
                  input_shape=(MAX_CHAR_LEN, char_vocab_size))(char_inputs)
    conv = MaxPooling1D(pool_size=3)(conv)

    conv5 = Flatten()(conv)
    chars = Dropout(0.5)(Dense(dense_outputs_chars, activation='relu')(conv5))

    words_input = Input(shape=(MAX_WORD_LEN,), dtype='int32', name='words_input')
    words = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                      trainable=False, name='emb_layer')(words_input)
    words = Flatten()(words)
    words = Dense(dense_outputs_words, activation='softmax', name='densed_emb_layer')(words)

    output = concatenate([chars, words])
    output = Dense(cat_output, activation='softmax', kernel_initializer='glorot_normal')(output)

    model = Model(inputs=[char_inputs, words_input], outputs=[output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    return model


def mini_batch_generator(x, y, w2i, batch_size=128):
    for i in range(0, len(x), batch_size):
        x_sample = x[i:i + batch_size]
        y_sample = y[i:i + batch_size]

        input_char_data = encode_char_data(x_sample)
        input_word_data = encode_word_data(w2i, x_sample)

        yield ([input_char_data, input_word_data], y_sample)


def encode_char_data(x):
    input_data = np.zeros((len(x), MAX_CHAR_LEN, char_vocab_size))
    for dix, sent in enumerate(x):
        counter = 0
        sent_array = np.zeros((MAX_CHAR_LEN, char_vocab_size))
        chars = list(sent.lower())
        for c in chars:
            if counter >= MAX_CHAR_LEN:
                pass
            else:
                char_array = np.zeros(char_vocab_size, dtype=np.int)
                if c in check:
                    ix = vocab[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        input_data[dix, :, :] = sent_array

    return input_data


def encode_word_data(w2i, x):
    input_data = np.zeros((len(x), MAX_WORD_LEN))
    for dix, post in enumerate(x):
        counter = 0
        post_array = np.zeros((MAX_WORD_LEN))
        words = post.split()
        for w in words:
            if counter >= MAX_WORD_LEN:
                pass
            else:
                ix = 0
                if w in w2i:
                    ix = w2i[w]
                post_array[counter] = ix
                counter += 1
        input_data[dix, :] = post_array

    return input_data


def shuffle_matrix(x, y):
    stacked = np.hstack((np.matrix(x).T, y))
    np.random.shuffle(stacked)
    xi = np.array(stacked[:, 0]).flatten()
    yi = np.array(stacked[:, 1:])

    return xi, yi


def get_next_block(get_data_neg, get_data_pos, get_data_neg_obf=None, get_data_pos_obf=None):
    X_neg, Y_neg = read_file(get_data_neg, False)
    X_pos, Y_pos = read_file(get_data_pos, True)

    if (get_data_neg_obf is not None and get_data_pos_obf is not None):
        X_neg_obf, Y_neg_obf = read_file(get_data_neg_obf, False)
        X_pos_obf, Y_pos_obf = read_file(get_data_pos_obf, True)
        X = X_neg
        X.extend(X_neg_obf)
        X.extend(X_pos)
        X.extend(X_pos_obf)
        Y = np.concatenate([Y_neg, Y_neg_obf, Y_pos, Y_pos_obf], axis=0)
    else:
        X = X_neg
        X.extend(X_pos)
        Y = np.concatenate([Y_neg, Y_pos], axis=0)

    X, Y = shuffle(X, Y, random_state=0)
    return X, Y


def generator(w2i, batch_size, get_data_neg, get_data_pos, get_data_neg_obf=None, get_data_pos_obf=None):
    X, Y = get_next_block(get_data_neg, get_data_pos, get_data_neg_obf, get_data_pos_obf)
    batches = mini_batch_generator(X, Y, w2i, batch_size=batch_size)

    while True:
        try:
            yield next(batches)
        except StopIteration:
            X, Y = get_next_block(get_data_neg, get_data_pos, get_data_neg_obf, get_data_pos_obf)
            batches = mini_batch_generator(X, Y, w2i, batch_size=batch_size)
            pass


def train_1ep_norm(model, gen, test_gen):
    model.fit_generator(gen, epochs=1, steps_per_epoch=BATCH_CNT, validation_data=test_gen,
                        validation_steps=TEST_BATCH_CNT, verbose=1)


def save_model(model, file):
    model_json = model.to_json()
    with open(file + '_json.json', "w") as json_file:
        json_file.write(model_json)

    model.save_weights(file + '_weights.h5')


def evaluate_on(model, neg_data, pos_data, w2i, show_mistakes=False):
    fpr = dict()
    tpr = dict()
    n_len = len(neg_data)
    o_len = len(pos_data)

    predict_on_test = neg_data
    predict_on_test.extend(pos_data)
    prediction = model.predict([encode_char_data(predict_on_test), encode_word_data(w2i, predict_on_test)])

    true_marks = []
    true_marks.extend(np.zeros(n_len))
    true_marks.extend(np.ones(o_len))

    fpr[0], tpr[0], _ = roc_curve(true_marks, prediction[:, 0])
    fpr[1], tpr[1], _ = roc_curve(true_marks, prediction[:, 1])
    prediction_score = prediction[:, 1]
    prediction = [np.argmax(marks) for marks in prediction]

    if show_mistakes:
        for i in range(len(predict_on_test)):
            if true_marks[i] != prediction[i]:
                print("Error: expected", true_marks[i], "got", prediction[i], predict_on_test[i])

    # plt.figure()
    # lw = 2
    # plt.plot(fpr[1], tpr[1], color='darkorange',
    #          lw=lw, label='ROC curve')
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

    roc_auc = roc_auc_score(true_marks, prediction_score)
    f1 = f1_score(true_marks, prediction, average='binary')
    acc = accuracy_score(true_marks, prediction)
    print("f1", f1, "acc", acc, "rocauc", roc_auc)


def load_w2v_vocab(w2v_vocab_path):
    with open(w2v_vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data

    return word2idx


def get_predictions(model, w2i, test_texts):
    return model.predict([encode_char_data(test_texts), encode_word_data(w2i, test_texts)])


if __name__ == '__main__':
    params = {}
    for line in open("config.txt", "r"):
        param = line.split("=")
        params[param[0]] = param[1]

    negative_file_path = params["neg"].strip()
    positive_file_path = params["pos"].strip()
    negative_obf_file_path = params["neg_obf"].strip()
    positive_obf_file_path = params["pos_obf"].strip()

    negative_test_file_path = params["test_neg"].strip()
    positive_test_file_path = params["test_pos"].strip()

    w2v_path = params["w2v"].strip()

    epochs = 1
    if params.__contains__("epoch"):
        epochs = int(params["epoch"].strip())

    model_w2v = Word2Vec.load(w2v_path)

    embedding_matrix = np.zeros((len(model_w2v.wv.vocab), 300))
    for i in range(len(model_w2v.wv.vocab)):
        embedding_vector = model_w2v.wv[model_w2v.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = make_model(embedding_matrix)
    # model = load_model('model.h5')

    w2i = {token: token_index for token_index, token in enumerate(model_w2v.wv.index2word)}

    use_obf = params['train_with_obf'].strip() == 'true'

    if (use_obf):
        gen = generator(w2i, BATCH_SIZE, get_negative_data, get_positive_no_hu_data, get_negative_obf_data,
                        get_positive_no_hu_obf_data)
    else:
        gen = generator(w2i, BATCH_SIZE, get_negative_data, get_positive_no_hu_data)

    test_gen = generator(w2i, BATCH_SIZE, get_negative_data, get_positive_no_hu_data, get_negative_obf_data,
                         get_positive_no_hu_obf_data)

    for i in range(epochs):
        train_1ep_norm(model, gen, test_gen)
        model.save('model.h5')

    evaluate_on(model, getData(negative_test_file_path)[:EVAL_SET_SIZE], getData(positive_test_file_path)[:EVAL_SET_SIZE],
                w2i)
