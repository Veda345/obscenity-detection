import os

from util import get_vocab, write_train_test_vocabs, prepare_file, do_dirty_labeling_for_file
from obfuscation import obfuscate_file


def separate_vocab(vocab_path, roots):
    vocab_name = os.path.splitext(vocab_path)[0]
    vocab = get_vocab(vocab_path)
    print("Got vocabulary with {:d} elements.".format(len(vocab)))

    write_train_test_vocabs(vocab, roots, vocab_name)
    print("Divided vocabulary for train and test.")


def preprocess_data(file_path):
    prepared_file_path = os.path.splitext(file_path)[0] + "_preprocessed"
    prepare_file(file_path, prepared_file_path)
    print("Pre-processed initial dataset.")
    return prepared_file_path


def label_data_with_vocab(vocab_path, file_path):
    vocab = get_vocab(vocab_path)
    files = do_dirty_labeling_for_file(file_path, os.path.splitext(file_path)[0], vocab)
    print("Divided dataset into train and test.")
    return files


def obfuscate_data(file_path):
    obf_file_path = os.path.splitext(file_path)[0] + "_obf"
    obfuscate_file(file_path, obf_file_path)
    print("Obfuscated file.")


if __name__ == '__main__':
    params = {}
    for line in open("transformation_config.txt", "r"):
        param = line.split("=")
        if len(param) == 2:
            params[param[0]] = param[1]

    vocab_path = params["vocab"].strip()
    vocab_roots = params["vocab_test_roots"].strip().split(',')
    init_file_path = params["init_data"].strip()

    separate_vocab(vocab_path, vocab_roots)

    processed_file_path = preprocess_data(init_file_path)

    files = label_data_with_vocab(vocab_path, processed_file_path)

    obfuscate_data(files[0])
    obfuscate_data(files[1])

    # obfuscate_data(init_file_path)

