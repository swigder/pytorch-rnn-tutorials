import time

import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from names.data import all_letters, EOS
from names.generator.data import random_training_set, make_category_input, make_chars_input


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_categories):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


def train(rnn, criterion, optimizer, category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn.init_hidden()
    optimizer.zero_grad()
    loss = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])

    loss.backward()
    optimizer.step()

    return output, loss.data[0] / input_line_tensor.size()[0]


def train_all_epochs(rnn, criterion, optimizer, categories, lines,
                     n_epochs=100000, print_every=5000, plot_every=1000):
    all_losses = []
    loss_avg = 0  # Zero every plot_every epochs to keep a running average

    def time_since(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    for epoch in range(1, n_epochs + 1):
        output, loss = train(rnn, criterion, optimizer, *random_training_set(categories, lines))
        loss_avg += loss

        if epoch % print_every == 0:
            print('%s (%d %d%%) %.4f' % (time_since(start), epoch, epoch / n_epochs * 100, loss))

        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0

    return all_losses


max_length = 20


# Generate given a category and starting letter
def generate_one(rnn, all_categories, category, start_char='A', temperature=0.5):
    category_input = make_category_input(category, all_categories)
    chars_input = make_chars_input(start_char)
    hidden = rnn.init_hidden()

    output_str = start_char

    for i in range(max_length):
        output, hidden = rnn(category_input, chars_input[0], hidden)

        # Sample as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Stop at EOS, or add to output_str
        if top_i == EOS:
            break
        else:
            char = all_letters[top_i]
            output_str += char
            chars_input = make_chars_input(char)

    return output_str

