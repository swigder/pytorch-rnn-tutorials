import torch
from torch.autograd import Variable

from names.classifier.data import read_data, n_letters, letter_to_tensor, line_to_tensor, random_training_pair
from names.classifier.rnn import RNN

all_categories, all_lines = read_data('../../data/names')
n_categories = len(all_categories)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

input = Variable(line_to_tensor('Albert'))
hidden = Variable(torch.zeros(1, n_hidden))

output, next_hidden = rnn(input[0], hidden)
print(output)

for i in range(10):
    category, line, category_tensor, line_tensor = random_training_pair(all_categories, all_lines)
    print('category =', category, '/ line =', line)
