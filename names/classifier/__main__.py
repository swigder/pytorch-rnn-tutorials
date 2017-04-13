import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

from names.classifier.rnn import RNN, train_all_epochs, confusion_matrix, predict
from names.data import read_data, n_letters

all_categories, all_lines = read_data('../../data/names')
n_categories = len(all_categories)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

criterion = torch.nn.NLLLoss()

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

all_losses = train_all_epochs(rnn, criterion, optimizer, all_categories, all_lines,
                              n_epochs=100000, print_every=5000, plot_every=1000)
plt.figure()
plt.plot(all_losses)

confusion = confusion_matrix(rnn, all_categories, all_lines)

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()


def predict_name(name):
    predict(rnn, all_categories, name)


while True:
    name = input("> ")
    predict_name(name)


