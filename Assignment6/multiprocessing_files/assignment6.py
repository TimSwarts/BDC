#!/usr/bin/env python3

"""Assignment 6: Ensemble Learning

This script creates so-calls 'bags' of data, using the MNIST dataset, by sampling with
replacement. It then trains a neural network on each 'bag'. These networks are then used
to create final predictions on the test set by using a majority voting system.
The predictions are then validated using the test set. This validation
is either shown or saved to a file, depending on user input.

Usage:
    assignment6.py

"""

__author__ = "Tim Swarts"
__version__ = "0.0.1"
__status__ = "Development"


import sys
import argparse
import time
import multiprocessing as mp
from typing import List, Tuple, Dict, Any
from pathlib import Path
from numpy.typing import NDArray
import numpy as np

MODEL_PATH = str(Path(__file__).parent.parent.joinpath("resources"))
sys.path.append(MODEL_PATH)

import data
import model


def init_args() -> argparse.Namespace:
    """Parse command line arguments.
    Returns:
        parser: A namespace containing the received arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train an ensemble of neural networks on the MNIST dataset."
    )

    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        metavar='',
        type=Path,
        default=Path("/students/2023-2024/Thema12/BDC_tswarts_372975/MNIST_mini.dat"),
        required=False,
        help=
            "Path to the directory containing the MNIST dataset.\
            Default is /students/2023-2024/Thema12/BDC_tswarts_372975/MNIST_mini.dat"
    )

    parser.add_argument(
        "-n",
        "--network_count",
        dest="network_count",
        metavar='',
        type=int,
        default=4,
        required=False,
        help="Number of networks to train. Default is 4."
    )

    parser.add_argument(
        "-s",
        "--data_size",
        dest="data_size",
        metavar='',
        type=int,
        default=5000,
        required=False,
        help="Number of instances to load from the dataset, max 60000. Default is 5000."
    )

    parser.add_argument(
        "-b",
        "--bag_size",
        dest="bag_size",
        metavar='',
        type=int,
        default=0,
        required=False,
        help=
            "Number of instances in each batch, if 0,\
            batch size wil equal the data size. Default is 0."
    )

    parser.add_argument(
        "-c",
        "--cores",
        dest="cores",
        metavar='',
        type=int,
        default=4,
        required=False,
        help="Number of cores to use for parallel processing. Default is 4."
    )

    parser.add_argument(
        "-t",
        "--training_ratio",
        dest="training_ratio",
        metavar='',
        type=float,
        default=0.8,
        required=False,
        help=
            "Ratio of the data to use for training remaining data is used for testing.\
            This value should range from 0.01 to 0.99. Default is 0.8."
    )


    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        metavar='',
        type=Path,
        default=None,
        required=False,
        help="Path to the output file. If none,\
            output is printed to the console. Default is None."
    )

    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        metavar='',
        type=int,
        default=30,
        required=False,
        help="Number of epochs to train each network for. Default is 30."
    )

    return parser.parse_args()


def parse_args() -> Tuple:
    """Further parse the command line arguments, handling default values and edge cases.
    Returns:
        args: A tuple of parsed arguments.
    """

    args = init_args()

    # Check if training ratio is within bounds
    if args.training_ratio > 0.95 or args.training_ratio < 0.01:
        print("Error: Training ratio must be float between 0.01 and 0.99.")
        sys.exit(1)

    # Check if the data size is within the bounds of the dataset
    if args.data_size > 60000:
        print("Error: Data size exceeds the number of instances in the dataset.")
        sys.exit(1)

    # Check if the batch size is within the bounds of the dataset
    if args.bag_size > args.data_size:
        print(
         "Warning: Batch size exceeds the number of instances in the training dataset."
        )

    # Set the batch size to the data size if it is 0
    if args.bag_size == 0:
        args.bag_size = args.data_size

    return args


def create_data_packages(
        x_data: np.ndarray[np.float64],
        y_data: np.ndarray[np.float64],
        bag_count: int,
        bag_size: int,
        epochs: int,
        test_xs: np.ndarray[np.float64]
    ) -> List[Tuple[Any]]:
    """Creates data packages for parallel network training. It creates the
    batches (bags) from the input data by sampling with replacement and
    adds the extra parameters needed for training

    Args:
        data: The dataset to sample from,
        a tuple containing an X (predictors) set and a y (response) variable.
        sample_count: The number of samples to generate.
        sample_size: The size of each sample.

    Returns:
        A tuple of of data bags.
    """

    indice_count = y_data.shape[0]
    packages = []

    for i in range(bag_count):
        # Create a sample of data indices of the given size, allowing duplicates
        indices = np.random.choice(range(indice_count), size=bag_size, replace=True)

        packages.append(
            (x_data[indices], y_data[indices], i + 1, epochs, test_xs)
        )

    return packages


def train_network(data_package: Tuple) -> Tuple[model.InputLayer, Dict]:
    """Train a neural network on the given data.

    Args:
        data: A data package containing the data to train the network on,
        the number of epochs to use for training. Each data packet thus contains
        four items:
            x_data,
            y_data,
            nework_number,
            number_of_epochs

    Returns:
        A tuple containing the trained network and the training history.
    """

    # The hyperparameters used are based on those used in my final assignment of the
    # Advanced Datamining course, but with alpha value increased for faster training
    alpha = 0.3

    x_data, y_data, nework_number, number_of_epochs = data_package

    # Create a neural network, this architecture is based on the one used in my final
    # assignment of the Advanced Datamining course
    network = (
        model.InputLayer(144, name=f"{nework_number}")
        + model.DenseLayer(72)
        + model.ActivationLayer(72, activation=model.hard_tanh)
        + model.DenseLayer(36)
        + model.ActivationLayer(36, activation=model.swish)
        + model.DenseLayer(18)
        + model.ActivationLayer(18, activation=model.swish)
        + model.DenseLayer(10)
        + model.SoftmaxLayer(10)
        + model.LossLayer(loss=model.categorical_crossentropy)
    )

    # Train the network
    history = network.fit(x_data, y_data, alpha=alpha, epochs=number_of_epochs)

    return network, history


def train_parallel_networks(
        data_bags: List[Tuple[np.ndarray[np.float64]]],
        cores: int
    ) -> List[Tuple[model.InputLayer, Dict]]:
    """Train a number of neural networks in parallel.

    Args:
        data: The data to train the networks on.
        cores: The number of cores to use.

    Returns:
        A list of trained networks.
    """

    with mp.Pool(cores) as pool:  # pylint: disable=no-member
        networks = pool.map(train_network, data_bags)

    return networks


def train_and_predict(data_package: Tuple):
    """Train a network on the given data and predict the output for the test data.

    Args:
        data_package: A data package containing the training and test data.

    Returns:
        The predictions of the network on the test data.
    """
    training_package = data_package[:-1]
    current_network = train_network(training_package)
    prediction_package = (current_network[0], data_package[-1])
    return single_model_predict(prediction_package)


def train_and_predict_parallel(
        data_packages: List[Tuple], cores
    ) -> List[NDArray[np.float64]]:
    """Train and predict in parallel.

    Args:
        data_packages: A list of data packages containing the training and test data.

    Returns:
        A list of predictions.
    """

    with mp.Pool(cores) as pool:
        predictions = pool.map(train_and_predict, data_packages)

    return predictions


def single_model_predict(
        prediction_data_package: Tuple[model.InputLayer, np.ndarray[np.float64]]
    ) -> List[NDArray[np.float64]]:
    """Predict the output of for the test data on the given neural network.

    Args:
        prediction_data_package: A tuple containing the network and the data to predict.

    Returns:
        The predictions of the network on the given data. list(np.array(float))
    """

    # Unpack the data package
    network, x_data = prediction_data_package

    # Predict the output
    yhats = network.predict(x_data)
    return yhats


def bootstrap_aggregate(
        all_network_outputs: List[List[NDArray[np.float64]]]
    ) -> List[NDArray[np.float64]]:
    """Calculate final predictions by aggregating the predictions of
    the individual networks through majority vote.

    Args:
        all_network_outputs: numpy arrays of class propabilities for each instance as
        predicted by each network.

    Returns:
        A list of final  predictions after aggregating by majority vote.
    """

    return np.average(all_network_outputs, axis=0)


def evaluate(
        ys_data: NDArray[np.float64],
        yhats: List[NDArray[np.float64]],
        output: Path = None
    ) -> None:
    """Evaluate the final predictions by comparing them to the real class labels and
    creating a confusion matrix with the data module, showing the accuracy of the
    predictions. The figure is shown when output is None, and saved to the specified
    location when output is a path.

    Args:
        ys_data: The actual class labels for the test set.
        yhats: The final predictions made by the bootstrap aggragated
        ensemble, a list of numpy arrays of class propabilities for each instance.
        number_of_inputs: The number of input values per instance, used in
        confusion matrix function.
        output: The path to the output file.
    """

    # Create a confusion matrix
    plt, accuracy = data.confusion(ys_data, yhats)

    # Save or show the confusion matrix
    if output:
        plt.savefig(output)
    else:
        plt.show()

    print(f"Accuracy = {accuracy*100.0:.1f}%")


def split_train_test(xs_data, ys_data, train_ratio):
    """Split the data into training and test sets.

    Args:
        xs_data: The input data.
        ys_data: The target data.
        train_ratio: The ratio of the data to use for training.

    Returns:
        A tuple containing the training and test data.
    """

    # Shuffle the data
    indices = np.arange(ys_data.shape[0])
    np.random.shuffle(indices)

    # Split the data
    split_index = int(ys_data.shape[0] * train_ratio)
    train_xs = xs_data[indices[:split_index]]
    train_ys = ys_data[indices[:split_index]]
    test_xs = xs_data[indices[split_index:]]
    test_ys = ys_data[indices[split_index:]]

    return train_xs, train_ys, test_xs, test_ys


def main():
    """Main function of the script, combining all the other functions to train
    an ensemble of neural networks in parallel on the MNIST dataset. Trained networks
    are then used to predict a test sets in parallel, after which the predictions are
    aggragated through majority voting. The final predictions resulting from this
    are evaluated with a confusion matrix, showing the accuracy of the ensemble.

    Returns:
        0 if the script runs successfully.
    """
    args = parse_args()

    print(f"Running with arguments:\n\t{args}\nLoading data...")
    # Load the MNIST dataset
    xs_data, ys_data = data.mnist_mini(args.file, num=args.data_size)

    print("Data loaded, converting to numpy arrays...")
    # Turn the data into numpy arrays
    xs_data = np.array(xs_data)
    ys_data = np.array(ys_data)

    print(f"Data converted, splitting off {args.training_ratio * 100}% for training...")
    # Get the training and test sets
    train_xs, train_ys, test_xs, test_ys = split_train_test(
        xs_data,
        ys_data,
        args.training_ratio
    )

    print(f"Data split, creating {args.network_count} data packages for training...")
    # Create data packages for parallel training and predicting
    data_packages = create_data_packages(
        x_data=train_xs,
        y_data=train_ys,
        bag_count=args.network_count,
        bag_size=args.bag_size,
        epochs=args.epochs,
        test_xs=test_xs
    )


    print(
        f"Data packages created, training {args.network_count} networks in parallel..."
    )
    start_time = time.perf_counter()

    # Train the networks and get predictions
    predictions = train_and_predict_parallel(data_packages, args.cores)

    print(
        f"Networks finished training and sending their test data predictions in "
        f"{(time.perf_counter() - start_time) / 60.0:.1f} minutes "
        f"now aggregating predictions through majority voting..."
    )
    # Aggregate the predictions
    final_predictions = bootstrap_aggregate(predictions)

    print("Predictions aggregated, evaluating ensemble performance...")
    evaluate(
        ys_data=test_ys,
        yhats=final_predictions,
        output=args.output
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
