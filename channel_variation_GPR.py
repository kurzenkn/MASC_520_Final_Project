"""This function assess the variation in flow rate from channel to channel in a microfluidic
system that split flow from two pressurized reservoirs into four channels."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk
import gpflow as gpf
from scipy.stats import linregress

#   Load flow rate measurement data
flow_measurements = pd.read_csv('MASC 520 Project Data.csv')
test_flow_measurements = pd.read_csv('MASC 520 Project Test Data.csv')
analytical_pred = pd.read_csv('MASC 520 Project Analytical Predictions.csv')

#   Calculate experimental flow rates from volume and time measurements
flow_measurements["Flow Rate (mL/min)"] = 60*flow_measurements["Volume (mL)"]/flow_measurements["Time (sec)"]
test_flow_measurements["Flow Rate (mL/min)"] = 60*test_flow_measurements["Volume (mL)"]/test_flow_measurements["Time (sec)"]

#   Index the data for each channel
channel_1_data = flow_measurements.loc[flow_measurements["Channel No."] == 1]
channel_2_data = flow_measurements.loc[flow_measurements["Channel No."] == 2]
channel_3_data = flow_measurements.loc[flow_measurements["Channel No."] == 3]
channel_4_data = flow_measurements.loc[flow_measurements["Channel No."] == 4]


def get_GPR_eqs():
    """Get GPR fits of the flow rate against input pressure for each of the channels,
     all four channels together, and channels 2-4 together."""


def get_lin_reg_eqs():
    """Get linear regression fits of the flow rate against input pressure for each of the channels,
     all four channels together, and channels 2-4 together."""


def channel_clustering():
    """Analyze clustering of the data points. This function is a maybe."""


def calc_total_flow_correction():
    """Fit the correction to the Hagen-Poiseuille model for the total flow rate of the
    microfluidic system."""


def plot_GPR_results():
    """Plot the results of the GPR fit."""


def plot_lin_regress_results():
    """Plot the results of the linear regression fit."""


def plot_clustering():
    """Plot the clustering results."""


def plot_comparison():
    """Plot a comparison of the GPR and linear regression fits."""


def plot_correction():
    """Plot the correction to the Hagen-Poiseuille model for the total flow rate
    of the microfluidic system."""


def main():
    """Call functions to perform Gaussian Process Regression and plot results."""


if __name__ == "__main__":
    main()



