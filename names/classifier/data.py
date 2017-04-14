import random

import torch
from torch.autograd import Variable

from names.data import all_letters, n_letters


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
