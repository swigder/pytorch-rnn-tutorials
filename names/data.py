import glob
import unicodedata
import string
import torch
import random

from torch.autograd import Variable

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker
EOS = n_letters - 1


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def read_lines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


def read_data(base_dir):
    all_filenames = glob.glob(base_dir + '/*.txt')

    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    for filename in all_filenames:
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return all_categories, category_lines
