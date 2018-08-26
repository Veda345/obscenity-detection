import sys

from gensim.models import Word2Vec
import os
import numpy as np
import json


# Iterator takes 2 paths to directories containing positive and negative data and provides separate words from all files
#  from these directories
class DataFromDirectories(object):
    def __init__(self, dirname1, dirname2):
        self.dirname1 = dirname1
        self.dirname2 = dirname2
 
    def __iter__(self):
        for fname in os.listdir(self.dirname1):
            print("Reading...", fname)
            for line in open(os.path.join(self.dirname1, fname), encoding='utf-8', errors='ignore'):
                yield line.split()
                
        for fname in os.listdir(self.dirname2):
            print("Reading...", fname)
            for line in open(os.path.join(self.dirname2, fname), encoding='utf-8', errors='ignore'):
                yield line.split()


# Takes 1 path to dataset and provides separate words from it
class DataFromFile(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        print("Reading...", self.fname)
        for line in open(self.fname, encoding='utf-8', errors='ignore'):
            yield line.split()


def save_embeddings(path):
    model.save(path)

    # weights = model.wv.syn0
    # np.save(open(path + "_weights", 'wb'), weights)
    #
    # vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    # with open(path + "_vocab", 'w') as f:
    #     f.write(json.dumps(vocab))


if __name__ == '__main__':
    params = {}
    for line in open("transformation_config.txt", "r"):
        param = line.split("=")
        if len(param) == 2:
            params[param[0]] = param[1]

    file_path = params["init_data"].strip()

    sentences = DataFromFile(file_path)
    model = Word2Vec(sentences, size=300, window=5, min_count=20, iter=10)
    save_embeddings(os.path.splitext(file_path)[0] + "_w2v_model")

    # Or use pre-trained model
    # model = Word2Vec.load(file_path)

    print("Similar words example")
    print(model.similar_by_word('я', topn=15))
    print(model.similar_by_word('пидор', topn=15))
