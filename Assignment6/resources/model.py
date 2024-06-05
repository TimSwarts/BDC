#!/usr/bin/env python3

"""This model script contains functions and classes that can be used to create a
neural network model, based on the multi-layer perceptron (MLP) architecture.
Created models will be python objects that can be used to for training or evaluating
the model on a given dataset, or to make predictions with the trained model.

A typical way of creating a model with this library is through the concatenation of
different layers, where each layer represents a different part of the network. The
layers can be added to the model by using the + operator, or by using the add method
of the first layer. All networks should start with an InputLayer and end with a LossLayer.
DenseLayers are typically paired with ActivationLayers to construct the hidden layers
of the network, or with either an Activation or Softmax layer, to form the output layer
of the network. For example, a simple network with one hidden layer of 100 neurons and
a softmax output layer with 10 outputs, for a data set with 144 input features
can be created as follows:

    >>> model = (
        InputLayer(144)
        + DenseLayer(100)
        + ActivationLayer(100, activation=relu)
        + DenseLayer(10)
        + SoftmaxLayer(10)
        + LossLayer(loss=categorical_crossentropy)
    )

This example uses a relu activation function, but more are available in this library.

This script was created as part of the "Advanced Datamining" course for the bachelor
of bioinformatics at the Instute for Life Science & Technology (ILST) at the Hanze
University of Applied Sciences Groningen.
"""

__author__ = "Tim Swarts"


from math import e, sqrt, log
import random
from collections import Counter
from copy import deepcopy
import sys
import numpy as np


# Activation functions
def linear(pre_activation_value):
    """
    Identity function
    :param pre_activation_value: preactivation value (float)
    :return pre_activation_value: postactivation value (float)
    """
    return pre_activation_value


def sign(pre_activation_value):
    """
    Signum function, this function gives -1 for all negative inputs,
    1 for all positive inputs, and 0 for 0.
    :param pre_activation_value: preactivation value (float)
    :return: post activation value (float)
    """
    if pre_activation_value > 0:
        return 1
    if pre_activation_value < 0:
        return -1
    return 0


def tanh(pre_activation_value):
    """
    Tangent hyperbolic function, this gives output valies between -1 and 1.
    :param pre_activation_value: preactivation value (float)
    :return: post activation value (float)
    """
    return np.tanh(pre_activation_value)


def hard_tanh(pre_activation_value):
    """
    Hard tangent hyperbolic function, this gives output valies between -1 and 1.
    :param pre_activation_value: preactivation value (float)
    :return: post activation value (float)
    """
    if pre_activation_value < -1:
        return -1
    if pre_activation_value > 1:
        return 1
    return pre_activation_value


def softsign(pre_activation_value):
    """
    Softsign function, this gives output valies between -1 and 1.
    :param pre_activation_value: preactivation value (float)
    :return: post activation value (float)
    """
    return pre_activation_value / (1 + abs(pre_activation_value))


def sigmoid(pre_activation_value):
    """Sigmoid function, this gives output values between 0 and 1.

    Args:
        pre_activation_value: preactivation value (float)

    Returns
        post activation value (float)
    """
    if pre_activation_value < 0:
        return np.exp(pre_activation_value) / (1 + np.exp(pre_activation_value))
    return 1 / (1 + np.exp(-pre_activation_value))


def softplus(pre_activation_value):
    """Softplus function, this is a smooth approximation to the ReLU function

    Args:
        pre_activation_value: preactivation value (float)

    Returns:
        post activation value (float)
    """
    try:
        return np.log(1 + np.exp(pre_activation_value))
    except OverflowError:
        return pre_activation_value


def relu(pre_activation_value):
    """
    ReLU function, 0 for all negative inputs, identity function for all postive inputs
    :param pre_activation_value: preactivation value (float)
    :return: post activation value (float)
    """
    return np.maximum(0, pre_activation_value)


def swish(pre_activation_value, *, beta=1):
    """Swish function, alsco called sigmoid linear unit.
    The Swish function is defined as f(x) = x * sigmoid(beta*x)
    where sigmoid is the logistic sigmoid function and beta is a parameter that
    can be set(usually 1) or trained (in rare cases).

    Args:
        pre_activation_value: preactivation value (float)
        beta: parameter to control the leaning of the function
        towards sigmoid or identity function. (float)

    Returns:
        post activation value (float)
    """
    return pre_activation_value * sigmoid(pre_activation_value * beta)


# One hot encoded activation functions
def softmax(instance_output):
    """
    This is the softmax function, it takes in a vector of numerical values
    returned by the hidden layers of the network (instance_output)
    and converts it into a probability distribution
    :param instance_output: a vector of numerical values (list[float] | tuple[float])
    :return y: a one-hot encoded probability distribution where each value
    represents the probability of a class associated with the index of that value.
    (list)
    """
    # Normalise instance_output to prevent ZeroDivisionError and OverflowError,
    # by subtracting the max from the vector
    instance_output = [value - max(instance_output) for value in instance_output]

    # Calculate the sum of e^hi for every value hi in the list instance_output,
    # used as the denominator in the softmax function
    denominator = sum(e**hi for hi in instance_output)

    # Apply softmax function to the list instance_output
    y_hats = [e**ho / denominator for ho in instance_output]

    # Return probability distribution
    return y_hats


# Loss functions
def mean_squared_error(yhat, y):
    """
    Mean squared loss function, calculates loss
    :param yhat: predicted value (float)
    :param y: real value (float)
    """
    return (yhat - y) ** 2


def mean_absolute_error(yhat, y):
    """
    Mean absolute loss function, calculates loss
    :param yhat: predicted value (float)
    :param y: real value (float)
    """
    return abs(yhat - y)


def hinge(yhat, y):
    """
    Hinge loss function, calculates loss
    :param yhat: predicted value (float)
    :param y: real value (float)
    :return: loss between 0 and 1 (float)
    """
    return max(1 - yhat * y, 0)  # max(1−𝑦̂ ⋅𝑦,0)


def binary_crossentropy(yhat, y, epsilon=0.0001):
    """
    Binary Cross Entropy loss function, calculates loss
    :param yhat: predicted value (float)
    :param y: real value (float)
    :param epsilon: minimum limit used for pseudo_log (float)
    :return: loss between 0 and 1 (float)
    """
    return -y * pseudo_log(yhat, epsilon) - (1 - y) * pseudo_log(1 - yhat, epsilon)


def categorical_crossentropy(yhat, y, epsilon=0.0001):
    """
    Categorical Cross Entropy loss function (-y * ln(yhat)), calculates loss
    :param yhat: predicted value (float)
    :param y: real value (float)
    :param epsilon: minimum limit used for pseudo_log (float)
    :return: loss between 0 and 1 (float)
    """
    return -y * pseudo_log(yhat, epsilon)


# Support functions
def pseudo_log(x, epsilon=0.001):
    """
    This function substitutes the log function in the cross entropy log functions,
    to prevent math domain errors. When the input of the log is smaller than a set
    minimum epsilon, a pseudo log is used instead. This prevents the cases where, due
    to underflow or during numerical differentiation, log(x) is attempted with x <= 0
    :param x: The input of the log (float | int)
    :param epsilon: The minimum limit (float)
    :return: log of x (float)
    """
    if x < epsilon:
        return log(epsilon) +  (x - epsilon)/epsilon
    return log(x)


def derivative(function, delta=0.001):
    """
    This function returns a function that calculates a numerical approximation
    of the slope in a point on the input function
    :param function: This is the function for which a derivative
    function is set up (function)
    :param delta: This is the delta used as the difference between two points
    to approximate the derivative (float)
    :return function: The derivative function of the input function (function)
    """
    # Create a function that calculates a numerical approximation of the slope in
    # a point on the given input function
    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)
    # Give it a distinct name
    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    # Return the wrapper function
    return wrapper_derivative


def shuffle_related_lists(list1, list2):
    """
    Shuffles two lists such that the elements in both lists maintain their original
    pairings. The output is a tuple containing the shuffled versions of the input lists.

    Example:
        input: list1 = [1, 2, 3], list2 = ['a', 'b', 'c']
        output: (list1_shuffled, list2_shuffled) = ([2, 1, 3], ['b', 'a', 'c'])

    :param list1: The first list to shuffle, should have the same length as list2 (list)
    :param list2: The second list to shuffle, should have the same length as list1 (list)
    :return: A tuple containing the shuffled versions of list1 and list2,
    maintaining their original pairings (tuple)
    """
    # Combine the two lists into a list of tuples, where each tuple
    # contains one element from each list
    combined = list(zip(list1, list2))
    # Shuffle the list of tuples, ensuring that each tuple remains intact
    random.shuffle(combined)
    # Return the unzipped lists
    return zip(*combined)


# Neural network layers:
class Layer:
    classcounter = Counter()

    def __init__(self, outputs, *, name=None, next=None):
        Layer.classcounter[f'{type(self)}'] += 1
        if name is None:
            # If no name is given, generate name based on the class name and a counter
            # e.g. Layer_1, Layer_2, etc.
            name = f'{type(self).__name__}_{Layer.classcounter[f"{type(self)}"]}'
        self.inputs = 0
        self.outputs = outputs
        self.name = name
        self.next = next

    def __repr__(self):
        text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def add(self, next):
        """
        This function adds layers to the network, a new layer is saved
        in the self.next variable of the current layer or in that of the next
        layer if self.next is not empty.
        :param next: A new layer that will be added at the end of the network.
        """
        if self.next is None:
            self.next = next
            next.set_inputs(self.outputs)
        else:
            self.next.add(next)

    def set_inputs(self, inputs):
        self.inputs = inputs

    # optional __add__ function to allow usage of + operator to add layers
    def __add__(self, next):
        result = deepcopy(self)
        result.add(deepcopy(next))
        return result

    def __getitem__(self, index):
        """
        This function makes the network iterable
        """
        if index in (0, self.name):
            return self
        if isinstance(index, int):
            if self.next is None:
                raise IndexError('Layer index out of range')
            return self.next[index - 1]
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]
        raise TypeError(
            f'Layer indices must be integers or strings, not {type(index).__name__}'
        )

    def __call__(self, xs, ys, alpha=None):
        raise NotImplementedError('Abstract __call__ method')


class InputLayer(Layer):

    def __repr__(self):
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def set_inputs(self, inputs):
        raise NotImplementedError(
            "An InputLayer itself can not receive inputs from previous layers,"
            "as it is always the first layer of a network."
        )

    def __call__(self, xs, ys=None, alpha=None):
        return self.next(xs, ys, alpha)

    def predict(self, xs):
        yhats, _, _ = self(xs)
        return yhats

    def evaluate(self, xs, ys):
        _, losses, _ = self(xs, ys)
        lmean = sum(losses) / len(losses)
        return lmean

    def partial_fit(self, xs, ys, *, alpha=0.001):
        # Update the weights and biases and save the loss
        _, losses, _ = self(xs, ys, alpha)
        lmean = sum(losses) / len(losses)
        return lmean

    def fit(self, xs, ys, *, epochs=1, alpha=0.001, validation_data=None):
        # Initialise loss history dict:
        history = {'loss': []}
        if validation_data is not None:
            history['val_loss'] = []

        if not isinstance(xs, np.ndarray) or not isinstance(ys, np.ndarray):
            xs = np.array(xs)
            ys = np.array(ys)

        # Train data and append history dictionary
        for i in range(epochs):
            # Train the network and save the mean loss of the epoch
            loss_of_epoch = self.partial_fit(xs, ys, alpha=alpha)
            history['loss'].append(loss_of_epoch)
            # If validation data is given, evaluate it and add it to history as well
            if validation_data is not None:
                history['val_loss'].append(
                    self.evaluate(validation_data[0], validation_data[1])
                )
            # Print the number and loss of the epoch
            print(
                f"\033[1;36;49m \t Network {self.name: >2} has been trained "
                f"for {i + 1: >2} epoch(s), loss: {loss_of_epoch:.2f} \033[0m"
            )
            sys.stdout.flush()
        # Return the loss history
        return history


class DenseLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None):
        super().__init__(outputs, name=name, next=next)
        # Set biases, one bias for every neuron (equal to the number of outputs)
        self.bias = np.zeros(self.outputs)

        # initialise weights (filled later in set_inputs method)
        self.weights = None

    def __repr__(self):
        text = f"DenseLayer(outputs={self.outputs}, name={repr(self.name)})"
        if self.next is not None:
            text += " + " + repr(self.next)
        return text

    def set_inputs(self, inputs: int):
        """Sets the number of inputs of this layer and, if not present yet,
        initialises random weights.
        """
        # Set inputs
        self.inputs = inputs

        # Generate uniform random weights for all neurons
        limit = sqrt(6 / (self.inputs + self.outputs))
        if self.weights is None:
            self.weights = np.random.uniform(
                -limit, limit, (self.outputs, self.inputs)
            )

    def __call__(self, xs: np.ndarray, ys=None, alpha=None):
        """
        xs should be a 2D numpy array where each row is an input instance with
        a number of values equal to self.inputs.
        """
        # Start empty gradient vector
        gxs = None

        # Calculate pre-activation values for all instances
        pre_activations = np.dot(xs, self.weights.T) + self.bias

        # Send to next layer, and collect its yhats, losses, and gas
        yhats, losses, gas = self.next(pre_activations, ys, alpha)

        if alpha:
            # Calculate gradient vectors for all instances
            gxs = np.dot(gas, self.weights)

            # Scale alpha with the number of instances to prevent the update from being too large
            scaled_alpha = alpha / xs.shape[0]

            # Update bias and weights per instance
            for x, gan in zip(xs, gas):
                update_size = scaled_alpha * gan
                self.bias -= update_size
                self.weights -= np.outer(update_size, x)

        return yhats, losses, gxs


class ActivationLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None, activation=linear):
        super().__init__(outputs, name=name, next=next)
        # Ensure that the activation function and its derivative are vectorized
        self.activation = np.vectorize(activation)
        self.activation_derivative = np.vectorize(derivative(activation))

    def __repr__(self):
        text = (
            f'ActivationLayer(outputs={self.outputs},name={self.name}, '
            f'activation={self.activation.__name__})'
        )
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, aa: np.ndarray, ys=None, alpha=None):
        # Start empty gradient vector
        gas = None

        # Calculate the activation value for all instances
        post_activation_value = self.activation(aa)

        # Send post_activation_value to the next layer and collect its yhats, losses, and gls
        yhats, losses, gls = self.next(post_activation_value, ys, alpha)

        if alpha:
            # Calculate gradients from the loss to the pre_activation value
            gas = self.activation_derivative(post_activation_value) * gls

        return yhats, losses, gas


class LossLayer(Layer):
    def __init__(self, loss=mean_squared_error, name=None):
        super().__init__(outputs=None, name=name)
        self.loss = loss

    def __repr__(self):
        text = f'LossLayer(loss={self.loss.__name__}, name={self.name})'
        return text

    def add(self, next):
        raise NotImplementedError(
            "It is not possible to add a layer to a LossLayer,"
            "since a network should always end with a single LossLayer"
        )

    def __call__(self, model_outputs, ys=None, alpha=None):
        # yhats is the output of the previous layer, because the loss layer is always last
        yhats = model_outputs
        # losses will be a list of losses for all outputs in yhats, starts at None
        losses = None
        # gls will be list of gradient vectors, one for each instance, with one value
        # for each output of the prev layer, starts at None
        gls = None
        if ys is not None:
            losses = []
            # For all instances calculate loss:
            for yhat, y in zip(yhats, ys):
                # Take sum of the loss of all outputs(number of outputs previous
                # layer=inputs this layer)
                ln = sum(self.loss(yhat[o], y[o]) for o in range(self.inputs))
                losses.append(ln)

            # If there is a learning rate
            if alpha:
                gls = []

                # Calculate the derivative of the loss function
                loss_derivative = derivative(self.loss)

                # Calculate a gradient vectors for all instances in yhats
                for yhat, y in zip(yhats, ys):
                    # Each instance can have multiple outputs, with the derivative of
                    # the loss function we calculate dl/dyhat
                    gln = [loss_derivative(yhat[o], y[o]) for o in range(self.inputs)]
                    gls.append(gln)
                gls = np.array(gls)
            losses = np.array(losses)
        return yhats, losses, gls


class SoftmaxLayer(Layer):
    def __repr__(self):
        text = f'SoftmaxLayer(outputs={self.outputs}, name={self.name})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, hidden_layers_outputs, ys=None, alpha=None):
        probs = []   # Final predictions
        ghs = None   # Set gradient from loss to hidden layers output (h) to None
        for instance_outputs in hidden_layers_outputs:
            prob = softmax(instance_outputs)   # Probability distribution for one instance
            probs.append(prob)  # Collect all instances

        probs = np.array(probs)

        # Send probabilities to the next layer and collect its yhats, losses, and gls
        yhats, losses, gls = self.next(probs, ys, alpha)

        if alpha:
            # Calculate gradients from the loss to last layer neuron output values of all instances
            ghs = []
            for yhat, gln in zip(yhats, gls):
                # Calculate the gradient vectors for every instance,
                # a vector contains the gradient from the loss to h
                ghn = [
                    sum(
                        gln[o] * yhat[o] * ((i==o) - yhat[i])
                        for o in range(self.outputs)
                    )
                    for i in range(self.inputs)
                ]  # This uses the derivative of the softmax function: yno*(𝛿io-yni)
                ghs.append(ghn)

            ghs = np.array(ghs)

        return yhats, losses, ghs


def main():
    """
    main function used for testing
    """
    return 0


if __name__ == "__main__":
    main()
