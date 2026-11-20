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
analytical_pred_test = pd.read_csv('MASC 520 Project Analytical Prediction Test.csv')
total_flow_rate = pd.read_csv('MASC 520 Project Total Flow Rate.csv')
total_flow_rate_test = pd.read_csv('MASC 520 Project Total Flow Test.csv')

#   Calculate experimental flow rates from volume and time measurements
flow_measurements["Flow Rate (mL/min)"] = 60*flow_measurements["Volume (mL)"]/flow_measurements["Time (sec)"]
test_flow_measurements["Flow Rate (mL/min)"] = 60*test_flow_measurements["Volume (mL)"]/test_flow_measurements["Time (sec)"]

#   Extract the flow rate (q) and pressure (p) data for each channel and convert to numpy array
q1 = flow_measurements.loc[flow_measurements["Channel No."]==1, ["Flow Rate (mL/min)"]].to_numpy()
p1 = flow_measurements.loc[flow_measurements["Channel No."]==1, ["Pressure (mbar)"]].to_numpy(dtype=np.float64)

q2 = flow_measurements.loc[flow_measurements["Channel No."]==2, ["Flow Rate (mL/min)"]].to_numpy()
p2 = flow_measurements.loc[flow_measurements["Channel No."]==2, ["Pressure (mbar)"]].to_numpy(dtype=np.float64)

q3 = flow_measurements.loc[flow_measurements["Channel No."]==3, ["Flow Rate (mL/min)"]].to_numpy()
p3 = flow_measurements.loc[flow_measurements["Channel No."]==3, ["Pressure (mbar)"]].to_numpy(dtype=np.float64)

q4 = flow_measurements.loc[flow_measurements["Channel No."]==4, ["Flow Rate (mL/min)"]].to_numpy()
p4 = flow_measurements.loc[flow_measurements["Channel No."]==4, ["Pressure (mbar)"]].to_numpy(dtype=np.float64)

q234 = np.vstack((q2, q3, q4))
p234 = np.vstack((p2, p3, p4))

q1_test = test_flow_measurements.loc[test_flow_measurements["Channel No."]==1, ["Flow Rate (mL/min)"]].to_numpy()
p1_test = test_flow_measurements.loc[test_flow_measurements["Channel No."]==1, ["Pressure (mbar)"]].to_numpy(dtype=np.float64)

q2_test = test_flow_measurements.loc[test_flow_measurements["Channel No."]==2, ["Flow Rate (mL/min)"]].to_numpy()
p2_test = test_flow_measurements.loc[test_flow_measurements["Channel No."]==2, ["Pressure (mbar)"]].to_numpy(dtype=np.float64)

q3_test = test_flow_measurements.loc[test_flow_measurements["Channel No."]==3, ["Flow Rate (mL/min)"]].to_numpy()
p3_test = test_flow_measurements.loc[test_flow_measurements["Channel No."]==3, ["Pressure (mbar)"]].to_numpy(dtype=np.float64)

q4_test = test_flow_measurements.loc[test_flow_measurements["Channel No."]==4, ["Flow Rate (mL/min)"]].to_numpy()
p4_test = test_flow_measurements.loc[test_flow_measurements["Channel No."]==4, ["Pressure (mbar)"]].to_numpy(dtype=np.float64)

q234_test = np.vstack((q2_test, q3_test, q4_test))
p234_test = np.vstack((p2_test, p3_test, p4_test))

#   Extract Data for Correction
q_analytical = analytical_pred[["Flow Rate (mL/min)"]].to_numpy()
p_analytical = analytical_pred[["Pressure (mbar)"]].to_numpy(dtype = np.float64)

q_analytical_test = analytical_pred_test[["Flow Rate (mL/min)"]].to_numpy()
p_analytical_test = analytical_pred_test[["Pressure (mbar)"]].to_numpy(dtype = np.float64)

q_tot = total_flow_rate[["Flow Rate (mL/min)"]].to_numpy()
p_tot = total_flow_rate[["Pressure (mbar)"]].to_numpy(dtype = np.float64)

q_tot_test = total_flow_rate_test[["Flow Rate (mL/min)"]].to_numpy()
p_tot_test = total_flow_rate_test[["Pressure (mbar)"]].to_numpy(dtype = np.float64)

def get_gpr_models(input_kernel):
    """Get GPR fits of the flow rate against input pressure for each of the channels,
     all four channels together, and channels 2-4 together."""
    #   Create Gaussian Process Regression models using the kernel passed to the function
    model_1 = gpf.models.GPR(
        (p1, q1),
        kernel = input_kernel
    )
    model_2 = gpf.models.GPR(
        (p2, q2),
        kernel= input_kernel
    )
    model_3 = gpf.models.GPR(
        (p3, q3),
        kernel= input_kernel
    )
    model_4 = gpf.models.GPR(
        (p4, q4),
        kernel= input_kernel
    )
    model_234 = gpf.models.GPR(
        (p234, q234),
        kernel= input_kernel
    )

    #   Train the models
    opt = gpf.optimizers.Scipy()
    opt.minimize(model_1.training_loss, model_1.trainable_variables)
    opt.minimize(model_2.training_loss, model_2.trainable_variables)
    opt.minimize(model_3.training_loss, model_3.trainable_variables)
    opt.minimize(model_4.training_loss, model_4.trainable_variables)
    opt.minimize(model_234.training_loss, model_234.trainable_variables)

    return model_1, model_2, model_3, model_4, model_234

def get_gpr_correction(input_kernel):
    """Get GPR fit of the correction term to the Hagen-Poiseuille model for total flow rate."""
    r = q_tot - q_analytical
    model_correction = gpf.models.GPR(
        (p_tot, r),
        kernel = input_kernel
    )

    #   Train the models
    opt = gpf.optimizers.Scipy()
    opt.minimize(model_correction.training_loss, model_correction.trainable_variables)

    return model_correction

def plot_experimental_flow_rates():
    """Plot the experimentally measured flow rates."""
    #   Plot results of all flow rate data
    fig_exdata, ax_exdata = plt.subplots()
    for channel, color, marker in [
        (1, "red", "x"),
        (2, "blue", "o"),
        (3, "green", "s"),
        (4, "orange", "^"),
    ]:
        gpr_data = flow_measurements.loc[flow_measurements["Channel No."] == channel]
        ax_exdata.scatter(gpr_data["Pressure (mbar)"], gpr_data["Flow Rate (mL/min)"],
                          color=color, marker=marker, alpha=0.5, label=f"Channel {channel}")

    ax_exdata.set(xlabel="Pressure (mbar)", ylabel="Flow Rate (mL/min)", title = "Experimentally Measured Flow Rates")
    ax_exdata.legend()
    ax_exdata.grid(True)
    fig_exdata.show()

def plot_gpr_results(gpr1, gpr2, gpr3, gpr4, gpr234, title):
    """Plot the results of the GPR fit."""
    #   Calculate predicted means and 95% confidence interval for Y
    p_plot = np.linspace(79, 121, 100)[:, None]

    q1_mean, q1_var = gpr1.predict_y(p_plot)
    q1_low = q1_mean - 1.96*np.sqrt(q1_var)
    q1_up = q1_mean + 1.96*np.sqrt(q1_var)

    q2_mean, q2_var = gpr2.predict_y(p_plot)
    q2_low = q2_mean - 1.96 * np.sqrt(q2_var)
    q2_up = q2_mean + 1.96 * np.sqrt(q2_var)

    q3_mean, q3_var = gpr3.predict_y(p_plot)
    q3_low = q3_mean - 1.96 * np.sqrt(q3_var)
    q3_up = q3_mean + 1.96 * np.sqrt(q3_var)

    q4_mean, q4_var = gpr4.predict_y(p_plot)
    q4_low = q4_mean - 1.96 * np.sqrt(q4_var)
    q4_up = q4_mean + 1.96 * np.sqrt(q4_var)

    q234_mean, q234_var = gpr234.predict_y(p_plot)
    q234_low = q234_mean - 1.96 * np.sqrt(q234_var)
    q234_up = q234_mean + 1.96 * np.sqrt(q234_var)

    #   Plot GPR fits and comparison of test points to the GPR fits
    fig_gpr, ax_gpr = plt.subplots(2, 2,
                                   figsize = (10,7), layout= "constrained")
    data_color = "black"
    test_color = "red"
    mean_color = "blue"
    var_color = "blue"
    alpha = 0.1

    ax_gpr[0,0].scatter(p1, q1, color = data_color, marker="x", label="Measured")
    ax_gpr[0, 0].scatter(p1_test, q1_test, color= test_color, marker="o", label="Test")
    ax_gpr[0,0].plot(p_plot, q1_mean, color = mean_color, marker = "None", label="Predicted Flow Rate")
    ax_gpr[0,0].plot(p_plot, q1_up, "--", color = var_color, marker = "None", label="95% Confidence")
    ax_gpr[0, 0].plot(p_plot, q1_low, "--", color=var_color, marker="None")
    ax_gpr[0,0].fill_between(p_plot[:,0], q1_low[:,0], q1_up[:,0], color =var_color, alpha= alpha)
    ax_gpr[0, 0].set(title = "Channel 1")

    ax_gpr[0, 1].scatter(p2, q2, color=data_color, marker="x")
    ax_gpr[0, 1].scatter(p2_test, q2_test, color=test_color, marker="o")
    ax_gpr[0, 1].plot(p_plot, q2_mean, color=mean_color, marker="None")
    ax_gpr[0, 1].plot(p_plot, q2_up, "--", color=var_color, marker="None")
    ax_gpr[0, 1].plot(p_plot, q2_low, "--", color=var_color, marker="None")
    ax_gpr[0, 1].fill_between(p_plot[:, 0], q2_low[:, 0], q2_up[:, 0], color=var_color, alpha=alpha)
    ax_gpr[0, 1].set(title="Channel 2")

    ax_gpr[1, 0].scatter(p3, q3, color=data_color, marker="x")
    ax_gpr[1, 0].scatter(p3_test, q3_test, color=test_color, marker="o")
    ax_gpr[1, 0].plot(p_plot, q3_mean, color=mean_color, marker="None")
    ax_gpr[1, 0].plot(p_plot, q3_up, "--", color=var_color, marker="None")
    ax_gpr[1, 0].plot(p_plot, q3_low, "--", color=var_color, marker="None")
    ax_gpr[1, 0].fill_between(p_plot[:, 0], q3_low[:, 0], q3_up[:, 0], color=var_color, alpha=alpha)
    ax_gpr[1, 0].set(title="Channel 3")

    ax_gpr[1, 1].scatter(p4, q4, color=data_color, marker="x")
    ax_gpr[1, 1].scatter(p4_test, q4_test, color=test_color, marker="o")
    ax_gpr[1, 1].plot(p_plot, q4_mean, color=mean_color, marker="None")
    ax_gpr[1, 1].plot(p_plot, q4_up, "--", color=var_color, marker="None")
    ax_gpr[1, 1].plot(p_plot, q4_low, "--", color=var_color, marker="None")
    ax_gpr[1, 1].fill_between(p_plot[:, 0], q4_low[:, 0], q4_up[:, 0], color=var_color, alpha=alpha)
    ax_gpr[1, 1].set(title="Channel 4")

    for ax in ax_gpr.flat:
        ax.set_xlabel("Pressure (mbar)")
        ax.set_ylabel("Flow Rate (mL/min)")
        ax.grid(True)

    fig_gpr.legend(loc = "outside right upper")
    fig_gpr.suptitle(title, fontsize=20)
    fig_gpr.show()

    fig_gpr234, ax_gpr234 = plt.subplots(figsize=(8,6), layout="constrained")
    ax_gpr234.scatter(p234, q234, color=data_color, marker="x", label="Measured")
    ax_gpr234.scatter(p234_test, q234_test, color=test_color, marker="o", label="Test")
    ax_gpr234.plot(p_plot, q234_mean, color=mean_color, marker="None", label="Predicted Flow Rate")
    ax_gpr234.plot(p_plot, q234_up, "--", color=var_color, marker="None", label="95% Confidence")
    ax_gpr234.plot(p_plot, q234_low, "--", color=var_color, marker="None")
    ax_gpr234.fill_between(p_plot[:, 0], q234_low[:, 0], q234_up[:, 0], color=var_color, alpha=alpha)

    ax_gpr234.set(xlabel="Pressure (mbar)", ylabel="Flow Rate (mL/min)", title=title+"\nChannels 2, 3, and 4")
    fig_gpr234.legend(loc = "outside right upper")
    ax_gpr234.grid(True)
    fig_gpr234.show()

def plot_correction(lin_corr, nonlin_corr, poly_corr):
    """Plot the correction to the Hagen-Poiseuille model for the total flow rate
    of the microfluidic system."""

    ### NOTE: Need to change plots of measured data to the residual

    #   Calculate predicted means and 95% confidence interval for Y
    p_plot = np.linspace(79, 121, 100)[:, None]

    lin_corr_mean, lin_corr_var = lin_corr.predict_y(p_plot)
    lin_corr_low = lin_corr_mean - 1.96 * np.sqrt(lin_corr_var)
    lin_corr_up = lin_corr_mean + 1.96 * np.sqrt(lin_corr_var)

    nonlin_corr_mean, nonlin_corr_var = nonlin_corr.predict_y(p_plot)
    nonlin_corr_low = nonlin_corr_mean - 1.96 * np.sqrt(nonlin_corr_var)
    nonlin_corr_up = nonlin_corr_mean + 1.96 * np.sqrt(nonlin_corr_var)

    poly_corr_mean, poly_corr_var = poly_corr.predict_y(p_plot)
    poly_corr_low = poly_corr_mean - 1.96 * np.sqrt(lin_corr_var)
    poly_corr_up = poly_corr_mean + 1.96 * np.sqrt(lin_corr_var)

    #   Plot GPR fits and comparison of test points to the GPR fits
    fig_corr, ax_corr = plt.subplots(3, 1,
                                   figsize=(10, 7), layout="constrained")
    data_color = "black"
    test_color = "red"
    mean_color = "blue"
    var_color = "blue"
    alpha = 0.1

    ax_corr[0].scatter(p_tot, q_tot-q_analytical, color=data_color, marker="x", label="Measured")
    ax_corr[0].scatter(p_tot_test, q_tot_test-q_analytical_test, color=test_color, marker="o", label="Test")
    ax_corr[0].plot(p_plot, lin_corr_mean, color=mean_color, marker="None", label="Predicted Flow Rate Correction")
    ax_corr[0].plot(p_plot, lin_corr_up, "--", color=var_color, marker="None", label="95% Confidence")
    ax_corr[0].plot(p_plot, lin_corr_low, "--", color=var_color, marker="None")
    ax_corr[0].fill_between(p_plot[:, 0], lin_corr_low[:, 0], lin_corr_up[:, 0], color=var_color, alpha=alpha)
    ax_corr[0].set(title="Linear + Bias Kernel Correction")

    ax_corr[1].scatter(p_tot, q_tot-q_analytical, color=data_color, marker="x")
    ax_corr[1].scatter(p_tot_test, q_tot_test-q_analytical_test, color=test_color, marker="o")
    ax_corr[1].plot(p_plot, nonlin_corr_mean, color=mean_color, marker="None")
    ax_corr[1].plot(p_plot, nonlin_corr_up, "--", color=var_color, marker="None")
    ax_corr[1].plot(p_plot, nonlin_corr_low, "--", color=var_color, marker="None")
    ax_corr[1].fill_between(p_plot[:, 0], nonlin_corr_low[:, 0], nonlin_corr_up[:, 0], color=var_color, alpha=alpha)
    ax_corr[1].set(title="Squared Exponential Kernel Correction")

    ax_corr[2].scatter(p_tot, q_tot-q_analytical, color=data_color, marker="x")
    ax_corr[2].scatter(p_tot_test, q_tot_test-q_analytical_test, color=test_color, marker="o")
    ax_corr[2].plot(p_plot, poly_corr_mean, color=mean_color, marker="None")
    ax_corr[2].plot(p_plot, poly_corr_up, "--", color=var_color, marker="None")
    ax_corr[2].plot(p_plot, poly_corr_low, "--", color=var_color, marker="None")
    ax_corr[2].fill_between(p_plot[:, 0], poly_corr_low[:, 0], poly_corr_up[:, 0], color=var_color, alpha=alpha)
    ax_corr[2].set(title="Polynomial Kernel Correction")

    for ax in ax_corr.flat:
        ax.set_xlabel("Pressure (mbar)")
        ax.set_ylabel("Flow Rate (mL/min)")
        ax.grid(True)

    fig_corr.legend(loc="outside right center")
    fig_corr.suptitle("Correction to Hagen-Poiseuille Model for Total Flow Rate", fontsize=20)
    fig_corr.show()

def get_lin_reg_eqs():
    """Get linear regression fits of the flow rate against input pressure for each of the channels,
     all four channels together, and channels 2-4 together."""


def plot_lin_regress_results():
    """Plot the results of the linear regression fit."""


def plot_fit_comparison():
    """Plot a comparison of the GPR and linear regression fits."""


def calc_model_validation_metrics():
    """Compute the metrics for model validation with the test points. Metrics include
    the coefficient of determination (R^2), normalized mean square error (NMSE),
    and mean log-likelihood (MLL)."""


def main():
    """Call functions to perform Gaussian Process Regression and plot results."""

    #   Plot experimental flow rate measurements
    plot_experimental_flow_rates()

    #   Call functions for Gaussian Process Regression for channels
    lin_gpr_1, lin_gpr_2, lin_gpr_3, lin_gpr_4, lin_gpr_234 = get_gpr_models(gpf.kernels.Linear() + gpf.kernels.Bias())
    nonlin_gpr_1, nonlin_gpr_2, nonlin_gpr_3, nonlin_gpr_4, nonlin_gpr_234 = get_gpr_models(gpf.kernels.SquaredExponential())
    poly_gpr_1, poly_gpr_2, poly_gpr_3, poly_gpr_4, poly_gpr_234 = get_gpr_models(gpf.kernels.Polynomial())
    plot_gpr_results(lin_gpr_1, lin_gpr_2, lin_gpr_3, lin_gpr_4, lin_gpr_234, "Linear + Bias Kernel")
    plot_gpr_results(nonlin_gpr_1, nonlin_gpr_2, nonlin_gpr_3, nonlin_gpr_4, nonlin_gpr_234, "Squared Exponential Kernel")
    plot_gpr_results(poly_gpr_1, poly_gpr_2, poly_gpr_3, poly_gpr_4, poly_gpr_234, "Polynomial Kernel")

    #   Call functions for Gaussian Process Regression for total flow rate correction
    lin_correction = get_gpr_correction(gpf.kernels.Linear() + gpf.kernels.Linear())
    nonlin_correction = get_gpr_correction(gpf.kernels.SquaredExponential())
    poly_correction = get_gpr_correction(gpf.kernels.Polynomial())
    plot_correction(lin_correction, nonlin_correction, poly_correction)


if __name__ == "__main__":
    main()



