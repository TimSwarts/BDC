"""
Suplementary functions for generating and visualizing datasets, provided by
Dr Dave R.M. Langers as part of the Advanced Dataming course,
modified by me to suit the needs of this BDC assignment.
"""

__author__ = "Dave Langers"

# IMPORTS:
import random
from math import pi, cos, sin, sqrt, floor
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from sklearn import metrics


# FUNCTIONS:
def mnist_mini(filename, num=60000, seed=None):
    """Returns a number of different random 12x12 MNIST images.

    Keyword arguments:
    filename -- full filename of the *.dat datafile
    num      -- number of images to randomly select (default 60000)
    seed     -- a seed to initialise the random number generator (default random)

    Return values:
    xs_data       -- 144-element lists of pixel values (range 0.0-1.0)
    ys_data       -- 10-element lists of correct digits using one-hot encoding
    """
    # Seed the random number generator
    random.seed(seed)
    # Initialise
    xs_data = []
    ys_data = []
    y_value = [0]*9 + [1]
    # Pick the digits
    with open(filename, 'rb') as datafile:
        for file_index in random.sample(range(6000), (num+9) // 10):
            datafile.seek(file_index*720)
            for _ in range(10):
                x_instance = []
                for byte in datafile.read(72):
                    x_instance.append((byte // 16 + random.random()) / 16.0)
                    x_instance.append((byte % 16 + random.random()) / 16.0)
                y_value = y_value[9:] + y_value[:9]
                xs_data.append(x_instance)
                ys_data.append(y_value)
    # Shuffle and return
    permutation = random.sample(range(len(xs_data)), num)
    return [xs_data[i] for i in permutation], [ys_data[i] for i in permutation]


def segments(classes, *, num=200, noise=0.0, seed=None):
    """Generate a dataset consisting of circular segments (i.e. "pizza slices").

    Arguments:
    classes  -- number of classes to generate

    Keyword options:
    num      -- number of instances (default 200)
    noise    -- the amount of noise to add (default 0.0)
    seed     -- a seed to initialise the random number generator (default random)

    Return values:
    xs_data       -- values of the attributes x1 and x2
    ys_data       -- class labels in one-hot encoding
    """

    # Seed the random number generator
    random.seed(seed)

    # Generate attribute data
    radii = [sqrt(3.0 * random.random()) for _ in range(num)]
    angles = [random.random() for _ in range(num)]
    xs_data = [
        [radius * cos(2.0 * pi * angle), radius * sin(2.0 * pi * angle)]
        for radius, angle in zip(radii, angles)
    ]
    ys_data = [
        [
            1.0 if floor(angle * classes) == class_label else 0.0
            for class_label in range(classes)
        ]
        for angle in angles
    ]

    # Add noise to the attributes
    for instance_index in range(num):
        for dimension_index in range(2):
            xs_data[instance_index][dimension_index] += random.gauss(0.0, noise)

    # Return values
    return xs_data, ys_data


def curve(series, title=None, save=False, filename=None):
    """Plots the curve of a given data series.

    Arguments:
    series   -- a dictionary of data series

    Return values:
    None
    """
    # Plot the curves and keep track of their x-range
    xmax = 1
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for series_index, label in enumerate(sorted(series.keys())):
        data = series[label]
        xmax = max(xmax, len(data))
        plt.plot(
            [x_val + 0.5 for x_val in range(len(data))],
            data,
            color=colors[series_index % len(colors)],
            linewidth=3.0,
            label=label
        )
        plt.axhline(
            y=min(data),
            color=colors[series_index % len(colors)],
            linewidth=1.0,
            linestyle='--'
        )
        plt.axhline(
            y=max(data),
            color=colors[series_index % len(colors)],
            linewidth=1.0,
            linestyle='--'
        )
    # Finish the layout
    plt.xlim([0, xmax])
    plt.legend()
    plt.grid(True, color='k', linestyle=':', linewidth=0.5)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=1.0)
    plt.xlabel(r'$n$')
    plt.ylabel(r'$y$')
    if title is not None:
        plt.title(title)
    if save:
        if filename is None:
            filename = title
        plt.savefig(f"{filename}.png")
    else:
        plt.show()


def digits(images, labels, model=None):
    """Shows 12x12 MNIST digit images with true (and predicted) labels.

    Keyword arguments:
    images  -- 144-element lists of pixel values (range 0-1)
    labels  -- 10-element lists of correct digits using one-hot encoding
    model   -- the classification model (default None)

    Return values:
    None
    """

    # Define the argmax helper function
    def argmax(list_values):
        max_value = -1.0
        result = -1
        for index, value in enumerate(list_values):
            if value > max_value:
                max_value = value
                result = index
        return result

    # Plot the digits
    num_images = len(images)
    _, axes = plt.subplots(
        1, num_images, figsize=(0.8 * num_images, 0.8), squeeze=False
    )
    for index, axis in enumerate(axes[0]):
        paint = [
            [images[index][row_index * 12 + column_index] for row_index in range(12)]
            for column_index in range(12)
        ]
        axis.imshow(
            paint, extent=(0.0, 1.0, 0.0, 1.0), vmin=0.0, vmax=1.0, cmap=plt.cm.binary
        )
        axis.set_aspect("equal", "box")
        axis.axis("off")
        title_text = f"{argmax(labels[index]):d}"
        if model is not None:
            title_text += f"→{argmax(model.predict([images[index]])[0]):d}"
        axis.set_title(title_text)
    plt.show()



def confusion(ys_data, yhats):
    """Shows 10x10 confusion matrix.

    Keyword arguments:
    xs_data       -- 144-element lists of pixel values (range 0-1)
    ys_data       -- 10-element lists of correct digits using one-hot encoding
    yhats    -- 10-element lists of predicted digits using one-hot encoding
    Return values:
    None
    """
    # # Define the argmax helper function
    # def argmax(ls):
    #     m = -1.0
    #     result = -1
    #     for n, l in enumerate(ls):
    #         if l > m:
    #             m = l
    #             result = n
    #     return result
    # Compute the confusion matrix
    matrix = metrics.confusion_matrix(
        [np.argmax(y) for y in ys_data],
        [np.argmax(yhat) for yhat in yhats],
        labels=list(range(10))
    )
    accuracy = sum(matrix[i][i] for i in range(10)) / ys_data.shape[0]
    # Plot the confusion matrix
    plt.imshow(matrix, norm=LogNorm(), cmap='Blues', origin='lower')
    plt.grid(True)
    plt.title(f'Accuracy = {accuracy*100.0:.1f}%')
    plt.xlabel('$\\hat{y}$')
    plt.ylabel('$y$')
    plt.xticks(range(10), list(range(10)))
    plt.yticks(range(10), list(range(10)))
    plt.colorbar()
    return plt, accuracy
