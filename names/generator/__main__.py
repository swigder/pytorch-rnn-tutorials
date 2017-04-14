import torch

from names.data import n_letters, read_data
from names.generator.rnn import RNN, train_all_epochs, generate_one

import matplotlib.pyplot as plt


all_categories, all_lines = read_data('../../data/names')
n_categories = len(all_categories)

learning_rate = 0.0005

rnn = RNN(n_letters, 128, n_letters, n_categories)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

all_losses = train_all_epochs(rnn, criterion, optimizer, all_categories, all_lines, plot_every=500)

plt.figure()
plt.plot(all_losses)

plt.show()


def generate(category, start_chars='ABC'):
    for start_char in start_chars:
        print(generate_one(rnn, all_categories, category, start_char))


while True:
    language, letters = input("> ").split()
    generate(language, letters)
