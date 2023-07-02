from math import e, sqrt, log, exp, tanh as m_tanh
import random
from collections import Counter
from copy import deepcopy
import time
import numpy as np
import data


# Activation functions
def linear(a):
    """
    Identity function
    :param a: preactivation value (float)
    :return a: postactivation value (float)
    """
    return a


def sign(a):
    """
    Signum function, this function gives -1 for all negative inputs, 1 for all positive inputs, and 0 for 0.
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    if a > 0:
        return 1
    if a < 0:
        return -1
    return 0


def tanh(a):
    """
    Tangent hyperbolic function, this gives output valies between -1 and 1.
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    return m_tanh(a)


def hard_tanh(a):
    """
    Hard tangent hyperbolic function, this gives output valies between -1 and 1.
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    if a < -1:
        return -1
    if a > 1:
        return 1
    return a


def softsign(a):
    """
    Softsign function, this gives output valies between -1 and 1.
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    return a / (1 + abs(a))


def sigmoid(a):
    """
    Sigmoid function, this gives output values between 0 and 1.
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    try:
        return exp(a) / (1 + exp(a))
    except OverflowError:
        return 1.0


def softplus(a):
    """
    Softplus function, this is a smooth approximation to the ReLU function
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    try:
        return log(1 + e**a)
    except OverflowError:
        return a


def relu(a):
    """
    ReLU function, 0 for all negative inputs, identity function for all postive inputs
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    return max(0, a)


def swish(a, *, beta=1):
    """
    Swish function, or sigmoid linear unit
    The Swish function is defined as f(x) = x * sigmoid(beta*x)
    where sigmoid is the logistic sigmoid function and beta is a parameter that can be set(usually 1)
    or trained(in rare cases).
    :param a: preactivation value (float)
    :param beta: parameter to control the leaning of the function towards sigmoid or identity function. (float)
    :return: post activation value (float)
    """
    return a * sigmoid(a * beta)


# One hot encoded activation functions
def softmax(h):
    """
    This is the softmax function, it takes in a vector of numerical values h and converts it into
    a probability distribution
    :param h: a vector of numerical values (list[float] | tuple[float])
    :return y: a one-hot encoded probability distribution where each value represents the probability of a class
               associated with the index of that value. (list)
    """
    # Normalise h to prevent ZeroDivisionError and OverflowError, by subtracting the max from the vector
    h = [value - max(h) for value in h]
    # Calculate the sum of e^hi for every value hi in the list h, used as the denominator in the softmax function
    denominator = sum(e**hi for hi in h)
    # Apply softmax function to the list h
    y_hats = [e**ho / denominator for ho in h]
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
    This function substitutes the log function in the cross entropy log functions, to prevent math domain errors.
    When the input of the log is smaller than a set minimum epsilon, a pseudo log is used instead. This prevents the
    cases where, due to underflow or during numerical differentiation, log(x) is attempted with x <= 0
    :param x: The input of the log (float | int)
    :param epsilon: The minimum limit (float)
    :return: log of x (float)
    """
    if x < epsilon:
        return log(epsilon) +  (x - epsilon)/epsilon
    return log(x)


def derivative(function, delta=0.001):
    """
    This function returns a function that calculates a numerical approximation of the slope in a point on the
    input function
    :param function: This is the function for which a derivative function is set up (function)
    :param delta: This is the delta used as the difference between two points to approximate the derivative (float)
    :return function: The derivative function of the input function (function)
    """
    # Create a function that calculates a numerical approximation of the slope in a point on the given input function
    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)
    # Give it a distinct name
    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    # Return the wrapper function
    return wrapper_derivative


def shuffle_related_lists(list1, list2):
    """
    Shuffles two lists such that the elements in both lists maintain their original pairings.
    The output is a tuple containing the shuffled versions of the input lists.

    Example:
        input: list1 = [1, 2, 3], list2 = ['a', 'b', 'c']
        output: (list1_shuffled, list2_shuffled) = ([2, 1, 3], ['b', 'a', 'c'])

    :param list1: The first list to shuffle, should have the same length as list2 (list)
    :param list2: The second list to shuffle, should have the same length as list1 (list)
    :return: A tuple containing the shuffled versions of list1 and list2, maintaining their original pairings (tuple)
    """
    # Combine the two lists into a list of tuples, where each tuple contains one element from each list
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
            name = f'{type(self).__name__}_{Layer.classcounter[f"{type(self)}"]}'  # example: Layer_1
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
        This function adds layers to the network, a new layer is saved in the self.next variable of the current layer
        or in that of the next layer if self.next is not empty.
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
        if index == 0 or index == self.name:
            return self
        if isinstance(index, int):
            if self.next is None:
                raise IndexError('Layer index out of range')
            return self.next[index - 1]
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]
        raise TypeError(f'Layer indices must be integers or strings, not {type(index).__name__}')

    def __call__(self, xs, ys, alpha=None):
        raise NotImplementedError('Abstract __call__ method')


class InputLayer(Layer):

    def __repr__(self):
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def set_inputs(self, inputs):
        raise NotImplementedError("An InputLayer itself can not receive inputs from previous layers,"
                                  "as it is always the first layer of a network.")

    def __call__(self, xs, ys=None, alpha=None):
        return self.next(xs, ys, alpha)

    def predict(self, xs):
        yhats, _, _ = self(xs)
        return yhats

    def evaluate(self, xs, ys):
        _, ls, _ = self(xs, ys)
        lmean = sum(ls) / len(ls)
        return lmean

    def partial_fit(self, xs, ys, *, alpha=0.001):

        # Update the weights and biases and save the loss
        _, ls, _ = self(xs, ys, alpha)
        lmean = sum(ls) / len(ls)
        return lmean


    def fit(self, xs, ys, *, epochs=1, alpha=0.001, validation_data=None):
        # Initialise loss history dict:
        history = {'loss': []}
        if validation_data is not None:
            history['val_loss'] = []

        # Save the start time
        start_time = time.time()

        # Train data and append history dictionary
        for i in range(epochs):
            # Train the network and save the mean loss of the epoch
            loss_of_epoch = self.partial_fit(xs, ys, alpha=alpha)
            history['loss'].append(loss_of_epoch)
            # If validation data is given, evaluate it and add it to history as well
            if validation_data is not None:
                history['val_loss'].append(self.evaluate(validation_data[0], validation_data[1]))

        # Return the loss history
        return history


class DenseLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None):
        super().__init__(outputs, name=name, next=next)
        # Set biases, one bias for every neuron (equal to the amount of outputs)
        self.bias = [0 for _ in range(self.outputs)]

        # Initialise weights (filled later in set_inputs method)
        self.weights = None

    def __repr__(self):
        text = f'DenseLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def set_inputs(self, inputs: int):
        self.inputs = inputs
        limit = sqrt(6 / (self.inputs + self.outputs))
        if not self.weights:
            self.weights = [[random.uniform(-limit, limit) for _ in range(self.inputs)] for _ in range(self.outputs)]

    def __call__(self, xs: list[list[float]], ys=None, alpha=None):
        """
        xs should be a list of lists of values, where each sublist has a number of values equal to self.inputs
        """
        aa = []   # Uitvoerwaarden voor alle instances xs (xs is een (nested) lijst met instances)
        gxs = None # Set gradient from loss to x-input equal to None
        for x in xs:
            a = []   # Uitvoerwaarde voor één instance x (x is een lijst met attributen)
            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                pre_activation = self.bias[o] + sum(self.weights[o][i] * x[i] for i in range(self.inputs))
                a.append(pre_activation)  # a is lijst met de output waarden van alle neuronen voor 1 instance
            aa.append(a)  # aa is een nested lijst met de output waarden van alle instances

        # Send to aa next layer, and collect its yhats, ls, and gas
        yhats, ls, gas = self.next(aa, ys, alpha)

        if alpha:
            # Initiate empty list
            gxs = []
            # Calculate gradient vectors for all instances
            for x, gan in zip(xs, gas):
                gxn = [sum(self.weights[o][i] * gan[o] for o in range(self.outputs)) for i in range(self.inputs)]
                # Add instance to list
                gxs.append(gxn)
                # Update bias and weights per instance
                for o in range(self.outputs):
                    # b <- b - alpha/N * xi * d_ln/d_ano (xi for bias = 1, therefore not included)
                    self.bias[o] = self.bias[o] - alpha/len(xs) * gan[o]
                    # w <- w - alpha/N * xi * d_ln/d_ano
                    self.weights[o] = [self.weights[o][i] - alpha/len(xs) * gan[o] * x[i] for i in range(self.inputs)]

        return yhats, ls, gxs


class ActivationLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None, activation=linear):
        super().__init__(outputs, name=name, next=next)
        self.activation = activation

    def __repr__(self):
        text = f'ActivationLayer(outputs={self.outputs}, name={self.name}, activation={self.activation.__name__})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, aa: list[list[float]], ys=None, alpha=None):
        hh = []   # Post activation values calculated from all pre activation values from the previous layer
        gas = None # Set gradient from loss to pre activation value to None
        for a in aa:
            h = []   # Post activation values of one instance
            for o in range(self.outputs):
                # Calculate post activation value for each neuron o, using the pre activation value (a)
                post_activation = self.activation(a[o])
                # Append post activation values of all neurons to data instance
                h.append(post_activation)
            # Collect all instances in the data set
            hh.append(h)

        # Send hh to the next layer and collect its yhats, ls, and gls
        yhats, ls, gls = self.next(hh, ys, alpha)

        if alpha:
            # Calculate gradients from the loss to the pre_activation value
            gas = []

            # Calculate activation derivative function
            activation_derivative = derivative(self.activation)
            # Calculate gradient vectors for all instances:, using the derivative of the activation function and gls
            for a, gln in zip(aa, gls):
                # For each instance, calculate the gradient from the loss to each pre activation value
                gan = [activation_derivative(a[i]) * gln[i] for i in range(self.inputs)]
                gas.append(gan)


        return yhats, ls, gas


class LossLayer(Layer):
    def __init__(self, loss=mean_squared_error, name=None):
        super().__init__(outputs=None, name=name)
        self.loss = loss

    def __repr__(self):
        text = f'LossLayer(loss={self.loss.__name__}, name={self.name})'
        return text

    def add(self, next):
        raise NotImplementedError("It is not possible to add a layer to a LossLayer,"
                                  "since a network should always end with a single LossLayer")

    def __call__(self, hh, ys=None, alpha=None):
        # yhats is the output of the previous layer, because the loss layer is always last
        yhats = hh
        # ls, the loss, which will be a list of losses for all outputs in yhats, starts at None
        ls = None
        # gls, will be list of gradient vectors, one for each instance, with one value for each output of the prev layer
        # starts None
        gls = None
        if ys is not None:
            ls = []
            # For all instances calculate loss:
            for yhat, y in zip(yhats, ys):
                # Take sum of the loss of all outputs(number of outputs previous layer=inputs this layer)
                ln = sum(self.loss(yhat[o], y[o]) for o in range(self.inputs))
                ls.append(ln)

            # If there is a learning rate
            if alpha:
                gls = []

                # Calculate the derivative of the loss function
                loss_derivative = derivative(self.loss)

                # Calculate a gradient vectors for all instances in yhats
                for yhat, y in zip(yhats, ys):
                    # Each instance can have multiple outputs, with the derivative of the loss we calculate dl/dyhat
                    gln = [loss_derivative(yhat[o], y[o]) for o in range(self.inputs)]
                    gls.append(gln)
        return yhats, ls, gls


class SoftmaxLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None):
        super().__init__(outputs, name=name, next=next)

    def __repr__(self):
        text = f'SoftmaxLayer(outputs={self.outputs}, name={self.name})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, hh, ys=None, alpha=None):
        probs = []   # Final predictions
        ghs = None   # Set gradient from loss to h value to None
        for h in hh:
            prob = softmax(h)   # Probability distribution for one instance
            probs.append(prob)  # Collect all instances

        # Send probabilities to the next layer and collect its yhats, ls, and gls
        yhats, ls, gls = self.next(probs, ys, alpha)

        if alpha:
            # Calculate gradients from the loss to last layer neuron output values of all instances
            ghs = []
            for yhat, gln in zip(yhats, gls):
                # Calculate the gradient vectors for every instance, a vector contains the gradient from the loss to h
                ghn = [sum(gln[o] * yhat[o] * ((i==o) - yhat[i]) for o in range(self.outputs))
                       for i in range(self.inputs)]   # This uses the derivative of the softmax function: yno*(𝛿io-yni)
                ghs.append(ghn)


        return yhats, ls, ghs


def main():
    """
    main function used for testing
    """

    return 0


if __name__ == "__main__":
    main()
