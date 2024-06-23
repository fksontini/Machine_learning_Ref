#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
names = ['Farrah', 'Fred', 'Felicia']
width = 0.5
fig, ax = plt.subplots()
ax.bar(names, fruit[0], width, label='Apple', color='red')
ax.bar(
    names,
    fruit[1],
    width,
    bottom=fruit[0],
    label='Bananas',
    color='yellow')
ax.bar(
    names,
    fruit[2],
    width,
    bottom=fruit[1] +
    fruit[0],
    label='Oranges',
    color='#ff8000')
ax.bar(
    names,
    fruit[3],
    width,
    bottom=fruit[2] +
    fruit[1] +
    fruit[0],
    label='Peaches',
    color='#ffe5b4')
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
plt.xticks(np.arange(0, 3, 1))
plt.yticks(np.arange(0, 81, 10))
ax.legend()
plt.show()
