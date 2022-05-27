# Do a plot of the training loss and accuracy over the iterations
# Use the training_log.csv file
# The beginning of an iteration is marked by the value of the epoch column being 0

import pandas as pd
import matplotlib.pyplot as plt

with open('./training_log.csv', 'r') as f:
    df = pd.read_csv(f)

# Count the number of iterations, i.e., the number of times there's a 0 in the epoch column
iterations = df['epoch'].value_counts()[0] + 1

epochs = {}  # iteration -> number of epochs

# For each iteration, count the number of epochs, and store it in the dictionary
n_epochs = 0
iteration = 0
for row in df.itertuples():
    if row.epoch == 0 and n_epochs != 0:
        epochs[iteration] = n_epochs
        n_epochs = 1
        iteration += 1
    else:
        n_epochs += 1

data = {}  # iteration -> training loss

current_iteration = 0
current_epoch = 0
rows = df.to_dict('index')

for i in rows:
    row = rows[i]
    if row['epoch'] == 0 and i != 0:
        current_iteration += 1
        current_epoch = 0
    else:
        try:
            data[current_iteration + current_epoch / epochs[current_iteration]] = {
                'loss': row['loss']
            }
        except KeyError:
            pass
        finally:
            current_epoch += 1

# plot the data and save the plot
iteration_list = list(data.keys())
loss_list = [data[i]['loss'] for i in iteration_list]

plt.plot(iteration_list, loss_list, label='loss')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.savefig('./training_loss.png')
plt.show()
