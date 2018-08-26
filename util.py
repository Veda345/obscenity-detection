import os
import re
import string


# The alphabet containing allowed characters
def get_alphabet():
    ru = ["а", "б", "в", "г", "д", "е", "ё", "ж", "з",
                 "и", "й", "к", "л", "м", "н", "о", "п", "р",
                 "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ",
                 "ъ", "ы", "ь", "э", "ю", "я"]
    alphabet = (ru + list(string.digits) +
                list(string.punctuation) + [' '] + list(string.ascii_lowercase))
    return alphabet


def get_vocab(file_name):
    f = open(file_name, "r")
    vocab = []
    for line in f:
        vocab.append(line.lower().strip())
    f.close()
    return set(vocab)


# Check text with given vocabulary
def contains_tokens(text, vocab):
    return any([term in vocab for term in text.split()])


def remove_br_tags(text):
    br_tags = re.compile('<br>|</br>')
    clean_text = re.sub(br_tags, '', text)
    return clean_text


# Remove <br> tags, lower the string and truncate to 1000 symbols
def prepare_file(init_file_path, result_file_path, max_symbol_len=1000):
    f_init = open(init_file_path, "r")
    f_res = open(result_file_path, "w")
    for line in f_init:
        line = line.lower()
        f_res.write(remove_br_tags(line[:max_symbol_len]))


# Roots are used for selecting test words. 
# Will create 2 files for new vocabs with prefix prefix_name and suffixes "_train" and "_test".
def write_train_test_vocabs(vocab, roots, prefix_name='obscene_vocab'):
    f_neg = open(prefix_name + "_train.txt", "w")
    f_pos = open(prefix_name + "_test.txt", "w")
    for word in vocab:
        if any(rooty in word for rooty in roots):
            f_pos.write(word + "\n")
        else:
            f_neg.write(word + "\n")


def do_dirty_labeling_for_file(file_path, prefix_name, vocab):
    pos_file_name = prefix_name + "_positive.txt"
    neg_file_name = prefix_name + "_negative.txt"
    with open(file_path, 'r') as f: 
        f_pos = open(pos_file_name, "w")
        f_neg = open(neg_file_name, "w")
        for line in f:
            if contains_tokens(line, vocab):
                f_pos.write(line)
            else:
                f_neg.write(line)
        f_pos.close()
        f_neg.close()
    return [neg_file_name, pos_file_name]


def getData(path):
    data = []
    f = open(path, "r")
    for line in f:
        data.append(line)
    f.close()
    return data


def create_char_vocab_set():
    alphabet = get_alphabet()
    vocab_size = len(alphabet)
    check = set(alphabet)
    alphabet = list(check)

    voc = {}
    reverse_voc = {}
    for ix, t in enumerate(alphabet):
        voc[t] = ix
        reverse_voc[ix] = t

    return voc, reverse_voc, vocab_size, check