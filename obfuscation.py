import os
import string
import random
import sys

from nltk.tokenize import TweetTokenizer


# The probability used for random obfuscation
fraction_of_obfuscations = 1/5

def tokenize(input):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(input)


# The alphabet containing allowed characters
def get_alphabet():
    ru = ["а", "б", "в", "г", "д", "е", "ё", "ж", "з",
                 "и", "й", "к", "л", "м", "н", "о", "п", "р",
                 "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ",
                 "ъ", "ы", "ь", "э", "ю", "я"]
    alphabet = (ru + list(string.digits) +
                list(string.punctuation) + [' '] + list(string.ascii_lowercase))
    return alphabet


# Maps letters to similar-looking digits
let2dig = {'a': ['4'], 'b': ['6'], 'c': ['7'], 'e': ['3'], 'g': ['6', '9'], 'i': ['1'], 'l': ['1'], 'o': ['0'],
           's': ['5'], 't': ['1', '7'], 'z': ['5', '2'],
           'а': ['4'], 'б': ['6', '8'], 'в': ['8', '13'], 'д': ['9'], 'е': ['3'], 'ё': ['3'], 'з': ['3'], 'о': ['0'],
           'т': ['7'], 'у': ['4'], 'ю': ['10'], 'я': ['91']}


def letter2digit(letter):
    if not letter in let2dig:
        return letter
    arr = let2dig[letter]
    ind = random.randint(0, len(arr)-1)
    return arr[ind]


# Read a map of homoglyphs
def get_homoglyphs():
    homoglyphs = {}
    f = open("data/homoglyphs.txt", "r")
    for line in f:
        frm = line[0]
        to = line[2]
        if frm in homoglyphs:
            homoglyphs[frm].append(to)
        else:
            homoglyphs[frm] = [to]
    f.close()
    homoglyphs['а'] = homoglyphs["a"]
    homoglyphs['е'] = homoglyphs["e"]
    homoglyphs['ё'] = homoglyphs["e"]
    homoglyphs['з'] = homoglyphs["3"]
    homoglyphs['к'] = homoglyphs["k"]
    homoglyphs['и'] = homoglyphs["u"]
    homoglyphs['н'] = homoglyphs["h"]
    homoglyphs['о'] = homoglyphs["o"]
    homoglyphs['р'] = homoglyphs["p"]
    homoglyphs['с'] = homoglyphs["c"]
    homoglyphs['т'] = homoglyphs["t"]
    homoglyphs['у'] = homoglyphs["y"]
    homoglyphs['х'] = homoglyphs["x"]
    homoglyphs['ч'] = homoglyphs["4"]
    homoglyphs['ь'] = homoglyphs["b"]
    return homoglyphs


# Maps letters to their homoglyphs
def letter2homoglyph(letter):
    homoglyphs = get_homoglyphs()
    if not letter in homoglyphs:
        return letter
    arr = homoglyphs[letter]
    ind = random.randint(0, len(arr)-1)
    return arr[ind]


def obfuscate_text(text, obfuscator):
    result = ''

    for i in range(len(text)):
        result += obfuscator(text, i)

    return result


# Random removal of characters
def delete_chars(text):
    obfuscator = lambda text, i: "" if random.randint(0, 100) < 100 * fraction_of_obfuscations else text[i]
    return obfuscate_text(text, obfuscator)


# Random insertion of characters
def insert_chars(text):
    alphabet = get_alphabet()
    obfuscator = lambda text, i: text[i] + alphabet[random.randint(0, len(alphabet)-1)] \
        if random.randint(0, 100) < 100 * fraction_of_obfuscations else text[i]
    return obfuscate_text(text, obfuscator)


# Random change of existing characters to characters from alphabet
def change_chars(text):
    alphabet = get_alphabet()
    obfuscator = lambda text, i: alphabet[random.randint(0, len(alphabet)-1)] \
        if random.randint(0, 100) < 100 * fraction_of_obfuscations else text[i]
    return obfuscate_text(text, obfuscator)


# Random change of characters to similar digits
def change_chars2digits(text):
    obfuscator = lambda text, i: letter2digit(text[i]) if random.randint(0, 100) < 100 * fraction_of_obfuscations else text[i]
    return obfuscate_text(text, obfuscator)


# Random change of characters to their homoglyphs
def change_chars2homoglyphs(text):
    homoglyphs = get_homoglyphs()
    obfuscator = lambda text, i: letter2homoglyph(text[i]) \
        if text[i] in homoglyphs and random.randint(0, 100) < 100 * fraction_of_obfuscations else text[i]
    return obfuscate_text(text, obfuscator)


# Random duplicate of characters
def duplicate_chars(text):
    obfuscator = lambda text, i: text[i] * random.randint(2, 10) if random.randint(0, 100) < 50 * fraction_of_obfuscations \
                                                                    and i < len(text)-1 else text[i]
    return obfuscate_text(text, obfuscator)


def obfuscate_file(frm, to):
    obfuscations = [delete_chars,
                    insert_chars,
                    change_chars,
                    change_chars2digits,
                    change_chars2homoglyphs,
                    duplicate_chars]
    f = open(frm, "r")
    f_to = open(to, "w")

    pos = 0
    for line in f:
        obfuscator = obfuscations[pos % len(obfuscations)]
        f_to.write(obfuscator(line))
        pos+=1