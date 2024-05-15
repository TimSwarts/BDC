#!/usr/bin/env python3

"""Assignment 6: Ensemble Learning

This script creates so-calls 'bags' of data, using the MNIST dataset, by sampling with
replacement. It then trains a neural network on each 'bag'. These networks are then
used to create final predictions on the test set by using a majority voting system.
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
import multiprocessing as mp
from typing import List, Tuple, Dict, Any
from pathlib import Path
from mpi4py import MPI
from numpy.typing import NDArray
import numpy as np

MODEL_PATH = str(Path(__file__).parent.parent.joinpath("resources"))
sys.path.append(MODEL_PATH)

import model
import data

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
HOST = MPI.Get_processor_name()


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
        type=Path,
        default=Path("/students/2023-2024/Thema12/BDC_tswarts_372975/MNIST_mini.dat"),
        required=False,
        help=
            "Path to the directory containing the MNIST dataset.\
            Default is /students/2023-2024/Thema12/BDC_tswarts_372975/MNIST_mini.dat."
    )

    parser.add_argument(
        "-n",
        "--network_count",
        dest="network_count",
        type=int,
        default=4,
        required=False,
        help="Number of networks to train. Default is 4."
    )

    parser.add_argument(
        "-s",
        "--data_size",
        dest="data_size",
        type=int,
        default=5000,
        required=False,
        help=
            "Number of instances to load from the dataset, max 60000. Default is 5000."
    )

    parser.add_argument(
        "-b",
        "--bag_size",
        dest="bag_size",
        type=int,
        default=0,
        required=False,
        help=
            "Number of instances in each batch, if 0,\
            batch size wil equal the data size. Default is 0."
    )


    parser.add_argument(
        "-t",
        "--training_ratio",
        dest="training_ratio",
        type=float,
        default=0.8,
        required=False,
        help=
            "Ratio of the data to use for training remaining data is used for testing.\
            Default is 0.8."
    )


    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=Path,
        default=None,
        required=False,
        help=
            "Path to the output file. If none, output is printed to the console.\
            Default is None."
    )

    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        type=int,
        default=30,
        required=False,
        help="Number of epochs to train each network for. Default is 30."
    )

    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        required=False,
        help=
            "Seed for the random number generator used for extracting MNIST data.\
            Default is None"
    )
    return parser.parse_args()


def parse_args() -> Tuple:
    """Further parse the command line arguments,
    handling default values and edge cases.
    Returns:
        args: A tuple of parsed arguments.
    """

    args = init_args()

    # Check if the data size is within the bounds of the dataset
    if args.data_size > 60000:
        print("Data size exceeds the number of instances in the dataset.")
        sys.exit(1)

    # Check if the batch size is within the bounds of the dataset
    if args.bag_size > args.data_size:
        print("Batch size exceeds the number of instances in the dataset.")
        sys.exit(1)

    # Set the batch size to the data size if it is 0
    if args.bag_size == 0:
        args.bag_size = args.data_size

    return args


def flatten_list(nested_list):
    """Flatten a nested list.

    Args:
        nested_list: A list of lists with any depth or format of nesting.
    
    Returns:
        A flat list containing all individual elements from the nested
        list as a single array of values (i.e).
    """
    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                flatten(item)
            else:
                flat_list.append(item)

    flat_list = []
    flatten(nested_list)
    return flat_list


def create_training_data_packages(
        x_data: np.ndarray[np.float64],
        y_data: np.ndarray[np.float64],
        bag_count: int,
        bag_size: int,
        epochs: int,
        validate: bool = False
    ) -> List[Tuple[Any]]:
    """Creates data packages for parallel network training. It creates the
    batches (bags) from the input data by sampling with replacement and
    adds the extra parameters needed for training, such as wehter to validate.

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

        packages.append((x_data[indices], y_data[indices], validate, i + 1, epochs))

    return packages


def train_network(training_data_package: Tuple) -> Tuple[model.InputLayer, Dict]:
    """Train a neural network on the given data.

    Args:
        data: A data package containing the data to train the network on, and a boolean
        indicating whether to validate the network. If validate is True, the data is
        split into training and validation data. Each data packet thus contains three
        items: x_data, y_data, and validate.

    Returns:
        A tuple containing the trained network and the training history.
    """

    # The hyperparameters used are based on those used in my final assignment of the
    # Advanced Datamining course, but with alpha value increased for faster training
    alpha = 0.3

    x_data, y_data, validate, nework_number, number_of_epochs = training_data_package

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
    if validate:
        x_train, y_train, x_val, y_val = split_train_test(x_data, y_data, 0.9)
        history = network.fit(
            x_train,
            y_train,
            alpha=alpha,
            epochs=number_of_epochs,
            validation_data=(x_val, y_val)
        )
    else:
        history = network.fit(x_data, y_data, alpha=alpha, epochs=number_of_epochs)

    return network, history


def single_model_predict(
        prediction_data_package: Tuple[model.InputLayer, np.ndarray[np.float64]]
    ) -> List[NDArray[np.float64]]:
    """Predict the output of for the test data on the given neural network.

    Args:
        prediction_data_package: A tuple containing the network and the data to
        predict.

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

    # Initialise vote tally and list of final predictions
    tally_per_instance = [np.zeros(10) for _ in range(len(all_network_outputs[0]))]
    final_predictions = np.copy(tally_per_instance)

    # Count the network votes for each instance
    for i, network in enumerate(all_network_outputs):
        for j, instance in enumerate(network):
            # Add a vote for the class with the highest probability
            tally_per_instance[j][np.argmax(instance)] += 1

    # Resolve the votes by setting class with the most votes to 1
    for i, instance in enumerate(tally_per_instance):
        final_predictions[i][np.argmax(instance)] = 1

    return final_predictions


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

    if RANK == 0:  # Controller/Root/Worker 0
        # Print some information about the run
        print(f"Running on {HOST} with {SIZE} processes.")
        print(f"Running with arguments:\n{args}")
        sys.stdout.flush()  # Flush buffer to show the print statements in slurm.out

        # Load the MNIST dataset
        xs_data, ys_data = data.mnist_mini(args.file, num=args.data_size)

        # Turn the data into numpy arrays
        xs_data = np.array(xs_data)
        ys_data = np.array(ys_data)

        print(
            f"Loaded {args.data_size} instances from the MNIST dataset.\n"
            f"Here is the first instance as an example:\n"
            f"X: {xs_data[:1]}\n"
            f"Y: {ys_data[:1]}\n"
        )
        sys.stdout.flush()
        # Get bags
        train_xs, train_ys, test_xs, test_ys = split_train_test(
            xs_data,
            ys_data,
            args.training_ratio
        )

        # Create data packages for parallel training
        training_data_packages = create_training_data_packages(
            x_data=train_xs,
            y_data=train_ys,
            bag_count=args.network_count,
            bag_size=args.bag_size,
            epochs=args.epochs
        )

        # Ensure even distribution of data packages
        train_packages_per_rank = [
            training_data_packages[i::SIZE] for i in range(SIZE)
        ]

    else: # Normal Workers
        train_packages_per_rank = None

    # Scatter the training data packages and fetch current rank's data package
    rank_training_packages = COMM.scatter(
        train_packages_per_rank, root=0
    )  # data is None for normal workers, so they only receive and don't send

    # Train network for current rank
    print(f"Rank {RANK} has received data, training network on training data...")
    sys.stdout.flush()
    rank_networks = []
    for rank_training_package in rank_training_packages:
        network, _ = train_network(rank_training_package)
        rank_networks.append(network)
    

    # Gather the results
    print(f"Sending/Gathering results in rank {RANK}...")
    nested_networks = COMM.gather(network, root=0)

    if RANK == 0:
        # Create data packages for parallel predictions
        print("Rank 0 is creating prediction data packages...")
        prediction_data_packages = [
            (network, test_xs) for network in flatten_list(nested_networks)
        ]

        # Ensure even distribution of data packages
        predict_packages_per_rank = [
            prediction_data_packages[i::SIZE] for i in range(SIZE)
        ]
    else:
        predict_packages_per_rank = None
    
    # Scatter the prediction data packages and fetch current rank's data package
    rank_prediction_package = COMM.scatter(
        predict_packages_per_rank, root=0
    )  # data is None for normal workers, so they only receive and don't send
    
    # Create predictions for current rank
    print(f"Rank {RANK} has received data, predicting test set...")
    rank_predictions = []
    for rank_prediction_package in rank_prediction_package:
        rank_predictions.append(single_model_predict(rank_prediction_package))
    
    # Gather the results
    print(f"Sending/Gathering results in rank {RANK}...")
    all_predictions = COMM.gather(rank_predictions, root=0)

    if RANK == 0:
        # Aggregate the predictions and evaluate
        print("Rank 0 is aggregating predictions...")
        flat_predictions = flatten_list(all_predictions)
        final_predictions = bootstrap_aggregate(flat_predictions)

        # Evaluate the final predictions
        print("Rank 0 is evaluating predictions...")
        evaluate(
            ys_data=test_ys,
            yhats=final_predictions,
            output=args.output
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
