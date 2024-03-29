# Assignment 6: Parallelization of Neural Nework training
> NB: This assignment resuses code from the Advanced Datamining course,
> my full code of which can be found here: [AdvancedDatamining](https://github.com/TimSwarts/AdvancedDatamining)
> IMPORTANT: This assingment is NOT finished yet, I will continue working on it after the deadline.
> PLEASE DO NOT GRADE YET!

## Changes made to the model script:
Some changes were made to the original model script, these changes are listed below and can be referenced
by taking a look at ``original_model.py`` and ``model.py``.

Changes:

* Since the batches are now divided on a more global level between the networks, this functionality
has been stripped from the partial_fit and fit method of InputLayer in the model script.
* The first few classes of the model script were remnant from earlier exercises in the Advanced Datamining
course and have been deleted.
* Backpropagation has been optimized:
The derivate of functions are now calculated outside of the
list comprehensions and loops that use them, to prevent the same calculations from being done multiple times.

## Changes made to the data script:
``data.py`` is a module delivered by Dave Langers as part of the Advanced Datamining course.
It contains functions to load and visualize data. The changes made to this script are listed below.

Changes:

* The ``curve`` function has been changed, it now includes options to add a title and to save the plot to a file.
This was done so that validation curves could be plotted for the different paralel models.
