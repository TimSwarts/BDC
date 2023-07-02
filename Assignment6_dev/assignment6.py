#!/usr/bin/env python3

"""
This script splits the MNIST dataset into smaller batches and trains a neural network on each batch.
These networks are then combined into an ensemble and tested on the test set.
"""

__author__ = "Tim Swarts"
__version__ = "0.0.1"
__status__ = "Development"


import argparse
import random
import multiprocessing as mp
from typing import List, Tuple
import time
import numpy as np
import data as dt
import model


def parse_args() -> argparse.Namespace:
    """
    Create an argument parser and return the parsed arguments.
    :return ap.Namespace: An object with the command line arguments.
    Use .[argument] to retrieve the arguments by name.
    """
    arg_parser = argparse.ArgumentParser(
        description="Train a neural network on the MNIST dataset."
    )

    arg_parser.add_argument(
        "-c",
        "--cores",
        type=int,
        default=8,
        metavar="",
        help="The number of cores to use."
             "This will be equal to the number of models trained. Default is 20.",
    )

    arg_parser.add_argument(
        "-n",
        "--number-of-instances",
        type=int,
        default=17600,
        metavar="",
        help="The number of MNIST instances to train on. Default is 17600.",
    )

    arg_parser.add_argument(
        "-e",
        "--epochs-per-model",
        type=int,
        default=40,
        metavar="",
        help="The number of epochs to train each model for. Default is 21.",
    )

    arg_parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=0.06,
        metavar="",
        help="The learning rate to use for training. Default is 0.06.",
    )

    arg_parser.add_argument(
        "--validate",
        action="store_true",
        help="Add this to validate each model on a validation dataset"
             " and show the validation curve. Default is False.",
    )

    return arg_parser.parse_args()



def train_network(data_tuple) -> model.InputLayer:
    """
    Train a neural network on the given data.
    :param data_tuple: A tuple containing:
                       the data, labels, epochs, learning rate, and name of the network,
                       and whether to validate the network.
    :return model.InputLayer: The trained neural network.
    """
    data, labels, epochs, learning_rate, name, validation_data = data_tuple

    # Create a neural network
    my_model = model.InputLayer(144, name=name) + \
           model.DenseLayer(72) + model.ActivationLayer(72, activation=model.hard_tanh) + \
           model.DenseLayer(36) + model.ActivationLayer(36, activation=model.swish) + \
           model.DenseLayer(18) + model.ActivationLayer(18, activation=model.swish) + \
           model.DenseLayer(10) + model.SoftmaxLayer(10) + \
           model.LossLayer(loss=model.categorical_crossentropy)

    start_time = time.time()

    my_history = my_model.fit(data,
                            labels,
                            alpha=learning_rate,
                            epochs=epochs)
    
    if validation_data:
        # Plot the validation curve
        dt.curve(
            my_history,
            title=name,
            save=True,
            filename=f"./output/{name}_validation_curve.png"
        )
        
    # Print the time it took to train the network
    print(f"Training {name} took {(time.time() - start_time)/60:.1f} minutes."
          f"{epochs} epochs trained with {learning_rate} learning rate.")
    # Return the trained networks
    return my_model


def parallel_training(data_tuples: List[Tuple], cores) -> List[model.InputLayer]:
    """
    Train multiple neural networks in parallel with Pool.map.
    :param data_tuples: A list of tuples containing the data, labels, epochs, learning rate,
                        and name of each network and whether to validate the network.
    :return result_networks: A list of trained neural networks. That need to be combined into an ensemble.
    """
    # Create a pool of processes
    with mp.Pool(processes=cores) as pool:
        # Use the pool to train the networks in parallel
        result_networks = pool.map(train_network, data_tuples)

    # Return the trained networks
    return result_networks


def combine_networks(networks: list[model.Layer], learning_rate: float) -> model.Layer:
    """
    Combine a list of neural networks into an ensemble.
    """
    # Create a new network with the same structure as the networks in the list
    final_network = model.InputLayer(144, name="final_network") + \
                    model.DenseLayer(72) + model.ActivationLayer(72, activation=model.hard_tanh) + \
                    model.DenseLayer(36) + model.ActivationLayer(36, activation=model.swish) + \
                    model.DenseLayer(18) + model.ActivationLayer(18, activation=model.swish) + \
                    model.DenseLayer(10) + model.SoftmaxLayer(10) + \
                    model.LossLayer(loss=model.categorical_crossentropy)

    # Average the weights of the networks in the list
    for i, layer in enumerate(final_network):
        if isinstance(layer, model.DenseLayer):
            final_network[i].weights = np.mean([network[i].weights for network in networks], axis=0)


    print(final_network[1].weights)

    # Average the biases of the networks in the list
    for i, layer in enumerate(final_network):
        if isinstance(layer, model.DenseLayer):
            final_network[i].bias = np.mean([network[i].bias for network in networks], axis=0)

    # Return the new network
    return final_network


def main():
    """
    This is the main function that executes all funciont calls in the correct order
    to eventually train an ensemble of neural networks to classify MNIST digits.
    :return: 0 (if the program ran successfully)
    """
    # Parse command line arguments (core count, epochs per network, learning rate, number of instances)
    args = parse_args()
    number_of_instances = args.number_of_instances
    cores = args.cores
    epochs = args.epochs_per_model
    learning_rate = args.learning_rate

    # Load the MNIST dataset
    xs_data, ys_data = dt.mnist_mini("./data/MNIST_mini.dat", num=args.number_of_instances)

    #Separate the data into training, validation, and test sets

    if args.validate:
        training_size = int(0.9 * args.number_of_instances)
        validation_size = int(0.05 * args.number_of_instances)

        trn_xs, trn_ys = xs_data[:training_size], ys_data[:training_size]
        val_xs, val_ys = (xs_data[training_size:training_size + validation_size],
                         ys_data[training_size:training_size + validation_size])
        test_xs, test_ys = (xs_data[training_size + validation_size:], ys_data[training_size + validation_size:])
        validation_data = (val_xs, val_ys)
    else:
        test_size = int(0.05 * args.number_of_instances)
        test_xs, test_ys = xs_data[:test_size], ys_data[:test_size]
        trn_xs, trn_ys = xs_data[test_size:], ys_data[test_size:]
        validation_data = None

    # Split the data into batches for parallel training
    data_batches = np.array_split(
        trn_xs, args.cores
    )  # list[np.ndarray[np.ndarray[float]]

    label_batches = np.array_split(
        trn_ys, args.cores
    )  # list[np.ndarray[np.ndarray[float]]

    data_tuples = [
        (
            data_batches[i],
            label_batches[i],
            epochs,
            learning_rate,
            f"model_{i}",
            validation_data,
        )
        for i in range(args.cores)
    ]

    # Train a neural network on each batch in parallel
    random.seed(0)
    networks = parallel_training(data_tuples, args.cores)


    # Combine the networks into an ensemble by averaging the weights
    ensemble = combine_networks(networks, learning_rate)

    # Evaluate the ensemble on the test set
    print(f'Loss: {ensemble.evaluate(test_xs, test_ys)}')

    # Show the confusion matrix
    dt.confusion(test_xs, test_ys, model=ensemble)

    return 0


if __name__ == "__main__":
    main()
