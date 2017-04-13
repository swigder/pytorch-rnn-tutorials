import glob
import unicodedata
import string
import torch
import random

from torch.autograd import Variable

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


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


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    letter_index = all_letters.find(letter)
    tensor[0][letter_index] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor[li][0][letter_index] = 1
    return tensor


def random_training_pair(categories, lines):
    category = random.choice(categories)
    line = random.choice(lines[category])
    category_tensor = Variable(torch.LongTensor([categories.index(category)]))
    line_tensor = Variable(line_to_tensor(line))
    return category, line, category_tensor, line_tensor

