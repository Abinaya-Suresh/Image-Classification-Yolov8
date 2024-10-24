
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the CSV file
results_path = r'C:\Users\abina\runs\classify\train4\results.csv'

# Read the CSV into a DataFrame
results = pd.read_csv(results_path)

# Strip any leading or trailing spaces from the column names
results.columns = results.columns.str.strip()

# Plot the training and validation loss
plt.figure()
plt.plot(results['epoch'], results['train/loss'], label='train loss')
plt.plot(results['epoch'], results['val/loss'], label='val loss', c='red')
plt.grid()
plt.title('Loss vs epochs')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()

# Plot the validation accuracy
plt.figure()
plt.plot(results['epoch'], results['metrics/accuracy_top1'] * 100)
plt.grid()
plt.title('Validation accuracy vs epochs')
plt.ylabel('accuracy (%)')
plt.xlabel('epochs')

# Display the plots
plt.show()
