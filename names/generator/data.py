import random

import torch
from torch.autograd import Variable

from names.data import all_letters, n_letters


# One-hot vector for category
def make_category_input(category, all_categories):
    li = all_categories.index(category)
    tensor = torch.zeros(1, len(all_categories))
    tensor[0][li] = 1
    return Variable(tensor)


# One-hot matrix of first to last letters (not including EOS) for input
def make_chars_input(chars):
    tensor = torch.zeros(len(chars), n_letters)
    for ci in range(len(chars)):
        char = chars[ci]
        tensor[ci][all_letters.find(char)] = 1
    tensor = tensor.view(-1, 1, n_letters)
    return Variable(tensor)


# LongTensor of second letter to end (EOS) for target
def make_target(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    tensor = torch.LongTensor(letter_indexes)
    return Variable(tensor)


# Get a random category and random line from that category
def random_training_pair(all_categories, all_lines):
    category = random.choice(all_categories)
    line = random.choice(all_lines[category])
    return category, line


# Make category, input, and target tensors from a random category, line pair
def random_training_set(all_categories, all_lines):
    category, line = random_training_pair(all_categories, all_lines)
    category_input = make_category_input(category, all_categories)
    line_input = make_chars_input(line)
    line_target = make_target(line)
    return category_input, line_input, line_target

