# Assignment 6: Parallelisation of Neural Nework training through bootsrap aggregation

This repository contains the final assignment for the Big Data Computing course, focusing on parallelising the training of a neural network. The original neural network module, created during the Advanced Datamining course, has been adapted to utilise a bootsrap aggregation (bagging) method for parallel model training. 

> NB: This assignment resuses code from the Advanced Datamining course,
> my full code of which can be found here: [AdvancedDatamining](https://github.com/TimSwarts/AdvancedDatamining)

## Introduction
### MNIST Numbers Dataset
The Modified National Institute of Standards and Technology (MNIST) database contains 60 thousand grayscale images of handwritten numbers from 0 to 9, originally with a resolution of 28 by 28 pixels. For the final assignment of the 2023 version of the Advanced Datamining course, Dave R.M. Langers miniaturised these images to 12 by 12 pixels, resulting in a dataset with 144 pixel intensity values per instance. In this dataset, each instance is paired with a corresponding one-hot encoded class label array specifying the number depicted. This data can be used to train neural networks for the classification of handwritten numbers. It was used here to test a bootstrap aggragation (bagging) setup for ensemble learning. I have made the data file available on the BIN network in a folder in the ``/students/`` directory, it can be found at:   
&ensp;&ensp;``/students/2023-2024/Thema12/BDC_tswarts_372975/MNIST_mini.dat``

### Bootstrap Aggregation
Bootstrap aggregation is an ensemble learning method in machine learning that involves the training of multiple models on different subsets of the training data. These subsets are created through sampling with replacement, resulting in diverse datasets where some instances are duplicated and others are ommitted. Each network is thus trained a different part of the data. The resulting models can then be used to make predictions on new data, and these predictions can be aggregated to make a final prediction. In our case, aggregation is done through soft voting based on the Softmax output of multiple neural networks. This means that each individual network calculates its own probability estimate for each instance, and these probabilities are than averaged to form the final prediction. The predicted number will be the number with the highest combined probability.

### Methods of Parallisation
This repository contains two versions of essentially the same bootstrap/aggregate script, they use different methods
of parallelising the training of the networks. The first version uses mpi4py, a library that enables distributed computing across multiple processors in a network, suitable for large-scale parallelisation. The second version employs the multiprocessing module with pool.map(), which leverages multiple cores on a single machine to achieve parallel execution. While mpi4py excels in scenarios requiring high scalability and inter-node communication, it can be complex to set up and manage. On the other hand, multiprocessing is simpler to implement for local parallel tasks but is limited to the resources of a single machine. These scripts provide a comparative look at these two parallelisation strategies.

### About the Scripts
Both python scripts (`assignment6.py` and `assigment6_mpi.py`) train a user-specified number of neural networks (with hard coded network architecture for simplicity) on a user-specified number of the MNIST number dataset for a user-specified number of epochs (complete network passes). They output the final prediction accuracy based on a test set split off from the training instances and either show a confusion matrix plot also containing the accuracy or save it to a png.

Both scripts have their own bash timing script that can be used to create a .csv file that shows how adding more parallel processes impacts the run time of the script.

> NB: If you want to change the timing scripts to test different parameters, it is best to keep
the `--number-of-networks` parameter above or equal to maximum number of cores/ranks; Adding more
cores than requested networks would be pointless, since there are only as many prallelisable tasks as there are networks.

## Changes to existing code
### Model Script (`model.py`)
Some changes were made to the original model module, these changes are listed below and can be referenced
by taking a look at [`original_model.py`](reference_material/original_model_comparison.py) and [`model.py`](resources/model.py).

Changes:

* The mini batch learning implementation has been removed from the from the partial_fit and fit method of InputLayer in the model script.
* The first few classes of the model script were remnants from earlier exercises in the Advanced Datamining course and have been deleted.
* Backpropagation has been optimized:
The derivative of functions are now calculated outside of the
list comprehensions and loops that use them, to prevent the same calculations from being done multiple times.
* Refactored the code to use numpy with vectorized activation and derivative functions to greatly speed up training. This also shortened the code a lot.
* The code style was improved to yield a higher pylint score, resulting in a lot of variables getting better, longer names.

### Data Script (`data.py`):
[`data.py`](resources/data.py) is a module delivered by Dave Langers as part of the Advanced Datamining course. It contains functions to load and visualize data. The changes made to this script are listed below.

Changes:

* The ``curve`` function has been changed, it now includes options to add a title and to save the plot to a file.
This was done so that validation curves could be plotted for different parallel models.
* The ``confusion`` function has been changed, it now takes in yhats directly instead of predicting them from a given model. It now also uses ``np.argmax`` to get the predicted classes from yhats, instead of the homemade function that was used before. Furthermore, it no longer shows the created confusion matrix, but instead returns it. This was done so that confusion matrices could be saved to a file. In addition to return the plot, the accuracy is the returned as well. 
* Code was refactored to fit pylint styling and redundant functions were removed, leaving only `segments`, `curve`, `mnist_mini` and `confusion`.

## Folder structure
The tree below cisualises the structure of this folder, it can be used as a reference to find files, both to know where executable scripts are and where to find output.

```bash
Assignment6/
├── plot_times.py                    # Script to plot timing results
├── assignment6.sh                   # Shell to run both timing scripts for assignment 6
├── README.md                        # This README file
├── multiprocessing_files/
│   ├── pool_map_timer.sh            # Timing script for multiprocessing version
│   └── assignment6.py               # Main script for multiprocessing version
├── mpi_files/
│   ├── slurm_script.sh              # SLURM script for running MPI version
│   ├── mpi_timer.sh                 # Timing script for MPI version
│   └── assignment6_mpi.py           # Main script for MPI version
├── reference_material/
│   ├── original_model_comparison.py # Script comparing original models
│   └── Eindopdracht_v2223.1.ipynb   # Jupyter notebook for the final Advanced Datamining assignment
├── resources/
│   ├── MNIST_mini.dat               # Dataset file for MNIST
│   ├── model.py                     # Script defining the model
│   └── data.py                      # Script handling data operations

```

## Usage
### Multiprocessing
All files needed to run the multiprocessing version of the bootstrap aggregation implementation can be found in the [`multiprocessing_files`](multiprocessing_files/) folder. All the parameters for the python script have been given default values and are thus optional, so that a simple run can be executed with:

```bash
python3 multiprocessing_files/assignment6.py
```

This will load 5000 MNIST instances, split off 80% of them for training (leaving the rest for testing), create 4 subsets of 5000 randomly sampled instances (with replacement), and train 4 networks for 30 epochs, using 4 cores. The script wil print the ensemble's accuracy on the test set and try to show the final confusion plot if possible. The script uses the data file located in the `/students/` folder (`/students/2023-2024/Thema12/BDC_tswarts_372975/MNIST_mini.dat`) by default.

Alternatively, file location, network count, data size, bag size, core count, training ratio, output file, and number of epochs can be set manually, using either their single letter representation (e.g. `-n` for network count) or full option name (e.g. `--network_count`). To get an explanation of each paramater, its defaults, and its possible values, you can run:

```bash
python3 multiprocessing_files/assignment6.py --help
```

Here's an example of specifying every parameter, this run uses

```bash
python3 ./multiprocessing_files/assignment6.py --file ./data/MNIST_mini.dat \
        --network_count 15 --data_size 15000 --bag_size 12000 --cores 15 \
        --training_ratio 0.9 --output ./output/confusion_matrix.png --epochs 30 \
```

To submit a timing job:
```bash
sbatch ./multiprocessing_files/pool_map_timer.sh
```
This will run python script with 15 networks and 15000 instances repeatedly with an increasing amount of cores (1 to 15). The standard output will be redirected to `./output/slurm_logs/pool_map_time_%j.out` and the standar error to `./output/slurm_logs/pool_map_time_%j.out`. Where `%j` will be replaced with the job ID.
The resulting csv file will be located at `./output/pool_timings.csv`.

### MPI
There are multiple ways to run the OpenMPI version of this bootstrap aggragate implementation. The first way is calling it with `mpiexec` directly. Just as before with the multiprocessing version, all argparse paramaters have defaults, so for a test run, they do not need to be provided. An important difference is however, that this script should be executed with `mpiexec` with a minimum of 2 ranks. This is because the script utilises a master/worker setup, where one rank (the master) divides data among the other ranks (the workers) who in turn do the actual work. The master rank eventually gathers and post-processes all the results.

A simple call could look like this:
```bash
mpiexec -np <number_of_ranks> python3 ./mpi_files/assignment6_mpi.py
# Where number_of_ranks >= 2
```

It can also be run with arguments, it takes exactly the same arguments as the multiprocessing version, except there's no `--cores` option.

Additionally, a slurm job for the `mpiexec` call can be submitted using [`/mpi_files/slurm_script.sh`](mpi_files/slurm_script.sh). Make sure to adjust the settings in this file to your preffered setup.

To submit a timing job:
```bash
sbatch ./mpi_files/mpi_timer.sh
```
This will run python script with 15 networks and 15000 instances repeatedly with an increasing amount of cores (1 to 15). The standard output will be redirected to `./output/slurm_logs/mpi_timer_%j.out` and the standar error to `./output/slurm_logs/mpi_timer_%j.out`. Where `%j` will be replaced with the job ID.
The resulting csv file will be located at: `./output/mpi_timings.csv`.