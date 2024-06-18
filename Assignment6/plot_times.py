#!/usr/bin/env python3

"""
This script is used to create a plot with two curves:
- One curve showing the time it takes to process data with pool.map()
  across different numbers of cores.
- One curve showing the time it takes to process data with OpenMPI
  across different numbers of ranks and tasks.
"""

import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def load_data():
    """
    Load the data from the two CSV files.
    :return: Tuple of two pandas.DataFrame objects.
    """
    pool_data = pd.read_csv("./output/pool_timings.csv", na_values=["NULL"])
    slurm_data = pd.read_csv("./output/mpi_timings.csv")
    return pool_data, slurm_data


def main():
    """
    Main function for the script.
    :return 0: Exit code 0 for success.
    """

    # Get dataframes
    pool_data, slurm_data = load_data()

    # Merge on number of cores and number of ranks
    merged_data = pd.merge(pool_data, slurm_data, left_on="Cores", right_on="Ranks")
    print("data:\n", merged_data)

    time_x = merged_data['Time_x']
    time_y = merged_data['Time_y']

    # Perform a paired t-test
    no_nan = merged_data.dropna()
    t_stat, p_value = stats.ttest_rel(no_nan['Time_x'], no_nan['Time_y'])
    x_mean = time_x.mean()
    y_mean = time_y.mean()
    mean_diff = x_mean - y_mean

    # Plot the data
    plt.plot(
        merged_data['Cores'],
        merged_data['Time_x'],
        label='pool.map()'
    )

    plt.plot(
        merged_data['Cores'],
        merged_data['Time_y'],
        label='OpenMPI'
    )

    plt.xlabel('Number of Cores/Ranks')
    plt.ylabel('Processing Time (s)')
    plt.title(
        'Plot of Processing Time vs Number of Cores/Ranks\n'
        ' for pool.map() and MPI scatter/gather'
    )
    plt.legend()
    plt.grid(True)

    # Add the extra text using plt.figtext
    plt.figtext(
        0.55, -0.03,  # Adjust y-coordinate as needed
        f'Difference in mean time (pool.map() - OpenMPI): {mean_diff:.2f}\n'
        f'Paired t-test results: t-statistic: {t_stat:.2f}, p-value: {p_value:.2e}',
        ha='center',  # Horizontal alignment
        fontsize=10,  # Adjust font size as needed
        wrap=True  # Enable text wrapping if needed
    )

    # Adjust layout to make space for the additional text
    plt.tight_layout(pad=2.0)

    plt.savefig('./output/timings_plot.png', bbox_inches='tight')  # Save with tight bounding box

    return 0


if __name__ == "__main__":
    sys.exit(main())
