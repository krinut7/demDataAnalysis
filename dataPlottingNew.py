"""PLot data collected from simulation and experiments."""

import string
import pandas as pd
from matplotlib import pyplot as plt
import sys
import seaborn as sns
import argparse

from dataPlotting import DataFrame


def exp_force(i: int) -> pd.DataFrame:
    """Read the experiment force data.

    Arguments:
        i (int): INdex for SR list
    Return:
        df (pd.DataFrame): df with exp_force values

    The axis for on wheel and estimator are different.
    Estimator is the conventional axis.
    Estimator -> On wheel: Fx -> -Fy, Fy -> -Fx, Fz -> -Fz
                           Mx -> -My, My -> -Mz, Mz -> -Mx
    - The columns are renamed according to the above convention.
    - Signs for Fy and Fz are changed.
    - Time is changed: string -> datetime -> seconds
    - Values for intitial 1 sec and last 2 secs are dropped.

    """
    data_type = "experimentData"
    csv_filename_onwheel = "leptrino_force_torque_on_wheel-force_torque.csv"
    csv_filename_inside = "leptrino_force_torque_center-force_torque.csv"

    filename_onwheel = (
        f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/2/{csv_filename_onwheel}"
    )
    filename_inside = (
        f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/2/{csv_filename_inside}"
    )

    df = pd.DataFrame()

    df_onwheel = pd.read_csv(
        filename_onwheel,
        usecols=[
            "time",
            ".wrench.force.x",
            ".wrench.force.y",
            ".wrench.force.z",
        ],
    )
    df_onwheel.columns = ["Time", "Fx", "Fy", "Fz"]

    df_inside = pd.read_csv(
        filename_inside,
        usecols=[
            "time",
            ".wrench.force.x",
            ".wrench.force.y",
            ".wrench.force.z",
        ],
    )
    df_inside.columns = ["Time", "Fx", "Fy", "Fz"]

    df_onwheel["Fy"] = -1 * df_onwheel["Fy"]
    df_onwheel["Fz"] = -1 * df_onwheel["Fz"]

    t1 = df_onwheel.loc[0, "Fy"] - df_inside.loc[0, "Fy"]  # for Fx
    t2 = df_onwheel.loc[0, "Fz"] - df_inside.loc[0, "Fx"]

    df["Fx"] = df_onwheel["Fy"] - t1
    df["Fz"] = df_onwheel["Fz"] - t2
    df["Time"] = df_onwheel["Time"]

    df["Fx/Fz"] = df["Fx"] / df["Fz"]
    df["Time"] = pd.to_datetime(df["Time"])
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    for x in df.index:
        df.loc[x, "Time"] = (
            df.loc[x, "Time"].seconds
            + df.loc[x, "Time"].microseconds / 1000000
        )

    df["Time"] = df["Time"] - df.loc[0, "Time"]
    df["Sample"] = range(0, len(df))
    df["Moving_Avg"] = df[PLOT_VALUE].ewm(span=100).mean()

    return df


def exp_sinkage(i: int) -> pd.DataFrame:
    """Read the experiment sinkage data.

    Arguments:
        i (int): INdex for SR list
    Return:
        df (pd.DataFrame): df with exp_sinkage values

    - Change time: string -> datetime -> seconds
    - Rename Columns.
    - Drop intial sec and last 2 seconds
    """
    data_type = "experimentData"
    csv_filename = "swt_driver-longitudinal_unit_log.csv"
    filename = f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{csv_filename}"

    df = pd.read_csv(filename, usecols=["time", ".traveling_distance"])
    df.columns = ["Time", "Sinkage"]

    df["Time"] = pd.to_datetime(df["Time"])
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    for x in df.index:
        df.loc[x, "Time"] = (
            df.loc[x, "Time"].seconds
            + df.loc[x, "Time"].microseconds / 1000000
        )

    df = df.drop(df[df.Time < 3].index)
    df = df.drop(df[df.Time > df.iloc[-1]["Time"] - 2].index).reset_index()
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    df["Sinkage"] = (df["Sinkage"] - df.loc[0, "Sinkage"]) / (10e6)
    df["Sample"] = range(0, len(df))
    df["Moving_Avg"] = df["Sinkage"].ewm(span=100).mean()

    return df


def exp_slip(i: int) -> pd.DataFrame:
    """Read and calculate the slip values for experiment.

    Arguments:
        i (int): INdex for SR list
    Return:
        df (pd.DataFrame): df with exp_slip values

    Algorithm:
        - Calulate slip by taking the values of conveying motor and wheel
         motor angular velocity.
        - Formula for slip: 1 - (omega_r)/v
        - drop intial 1 sec and last 2 secs
    """
    data_type = "experimentData"
    csv_filename_angular = "swt_driver-vertical_unit_log.csv"
    csv_filename_trans = "swt_driver-longitudinal_unit_log.csv"
    df = pd.DataFrame()

    filename_wheel = (
        f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{csv_filename_angular}"
    )
    filename_conveying = (
        f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{csv_filename_trans}"
    )

    df_wheel = pd.read_csv(
        filename_wheel, usecols=["time", ".wheel_motor_angular_vel"]
    )
    df_conveying = pd.read_csv(
        filename_conveying, usecols=["time", ".conveying_motor_angular_vel"]
    )

    df["Time"] = df_wheel["time"]
    df["Omega_conveying"] = df_conveying[".conveying_motor_angular_vel"]
    df["Omega_motor"] = df_wheel[".wheel_motor_angular_vel"]
    df["Slip"] = df["Omega_conveying"] / df["Omega_motor"]
    df["Slip"] = df["Slip"].fillna(df.loc[5, "Slip"])

    df["Time"] = pd.to_datetime(df["Time"])
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    for x in df.index:
        df.loc[x, "Time"] = (
            df.loc[x, "Time"].seconds
            + df.loc[x, "Time"].microseconds / 1000000
        )

    df = df.drop(df[df.Time < 1].index)
    df = df.drop(df[df.Time > df.iloc[-1]["Time"] - 2].index).reset_index()
    df["Time"] = df["Time"] - df.loc[0, "Time"]
    df["Sample"] = range(0, len(df))
    df["Moving_Avg"] = df["Slip"].ewm(span=100).mean()

    return df


def sim_force(i: int) -> pd.DataFrame:
    """Read the simulation force data.

    Arguments:
         i (int): INdex for SR list
    Return:
         df (pd.DataFrame): df with sim_force values)

    - First 2 seconds are dropped
    - Fx is divided by WHEEL_WEIGHT
    """
    data_type = "simulationData"
    csv_filename = "result_monitor.csv"
    filename = f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{csv_filename}"

    df = pd.read_csv(
        filename,
        header=1,
        usecols=["Time", "wheel.fx", "wheel.fy", "wheel.fz"],
    )
    df.columns = ["Time", "Fx", "Fy", "Fz"]

    df = df.drop(df[df.Time < 2].index).reset_index()

    df["Fz"] = df["Fz"] + WHEEL_WEIGHT
    df["Fx/Fz"] = df["Fx"] / df["Fz"]
    avg = abs(df["Fx"]).mean() / abs(df["Fz"]).mean()
    df["Fx/Fz"] = df["Fx/Fz"].fillna(avg)
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    df["Sample"] = range(0, len(df))
    df["Moving_Avg"] = df[PLOT_VALUE].ewm(span=100).mean()
    return df


def sim_sinkage(i: int) -> pd.DataFrame:
    """Read the simulation sinkae data.

    Arguments:
         i (int): INdex for SR list
    Return:
         df (pd.DataFrame): df with sim_sinkage values

    - First 2 secs are dopped.
    - Change sign of sinkage and subtract from the initial value.
    """
    data_type = "simulationData"
    csv_filename = "CenterOfMass.txt"
    filename = f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{csv_filename}"

    df = pd.read_csv(
        filename, sep=" ", names=["Time", "Sinkage"], skiprows=[0]
    )

    df = df.drop(df[df.Time < 2].index).reset_index()
    df["Time"] = df["Time"] - df.loc[0, "Time"]
    df["Sinkage"] = -1 * df["Sinkage"]
    df["Sinkage"] = df["Sinkage"] - df.loc[0, "Sinkage"]
    df["Sample"] = range(0, len(df))
    df["Moving_Avg"] = df["Sinkage"].ewm(span=100).mean()

    return df


def sim_slip(i: int) -> pd.DataFrame:
    """Read and calculate the slip for simulation."""
    data_type = "simulationData"
    csv_filename = "Velocity.txt"
    filename = f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{csv_filename}"

    df = pd.read_csv(
        filename,
        sep=" ",
        names=["Time", "Vx", "Vy", "Vz", "OmegaX", "OmegaY", "OmegaZ"],
        skiprows=[0],
    )

    df["Slip"] = 1 - (df["Vx"] / (WHEEL_RADIUS * df["OmegaY"]))
    df = df.drop(df[df.Time < 2].index).reset_index()
    df["Time"] = df["Time"] - df.loc[0, "Time"]
    df["Sample"] = range(0, len(df))
    df["Moving_Avg"] = df["Slip"].ewm(span=100).mean()

    return df


def force_slip(df: list) -> pd.DataFrame:
    """Calculate the average of force values for each slip."""
    df_force_avg = {"Slip": SR, "Force_Avg": list()}

    for df in df:
        df_force_avg["Force_Avg"].append(abs(df["Fx/Fz"]).mean())

    return pd.DataFrame(df_force_avg)


def plot_regression_line(df_fslip):
    """Plot regression line."""
    df1 = df_fslip["exp"]
    df2 = df_fslip["sim"]
    fig, ax = plt.subplots()
    plot1 = sns.regplot(
        x="Slip",
        y="Force_Avg",
        data=df1,
        fit_reg=True,
        order=2,
        ci=None,
        ax=ax,
        label="Experiment",
    )
    plot2 = sns.regplot(
        x="Slip",
        y="Force_Avg",
        data=df2,
        fit_reg=True,
        order=2,
        ci=None,
        ax=ax,
        label="Simulation",
    )
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.legend()
    plt.show()


def plot_data(
    i, df_exp: DataFrame, df_sim: DataFrame, plot_grid: tuple, yaxis_: string
):
    """Plot force for simulation and experiment."""
    AX[plot_grid].set(
        xlabel="Sample",
        ylabel=yaxis_,
        title=f"{yaxis_} SR:{SR}",
    )

    # plotting raw values
    """AX[plot_grid].plot(
        "Sample",
        yaxis_,
        data=df_exp,
        linestyle="-",
        label=f"Exp SR{SR[i]}",
    )
    AX[plot_grid].plot(
        "Sample",
        yaxis_,
        data=df_sim,
        linestyle="-",
        label=f"Sim SR{SR[i]}",
    )"""

    # plotting moving average
    AX[plot_grid].plot(
        "Sample",
        "Moving_Avg",
        data=df_exp,
        linestyle="-",
        label=f"Exp Moving Avg. SR{SR[i]}",
    )
    AX[plot_grid].plot(
        "Sample",
        "Moving_Avg",
        data=df_sim,
        linestyle="-",
        label=f"Sim Moving Avg. SR{SR[i]}",
    )

    AX[plot_grid].legend()


def main():
    """Call Main function.

    df_exp: dict for experiment with the dataframes for each slip ratio
    df_sim: dict for simulation with the dataframes for each slip ratio
    """
    for i in range(len(sys.argv) - 1):
        try:
            plot_data(i, exp_force(i), sim_force(i), (0, 0), PLOT_VALUE)
            plot_data(i, exp_sinkage(i), sim_sinkage(i), (0, 1), "Sinkage")
            plot_data(i, exp_slip(i), sim_slip(i), (1, 0), "Slip")

        except FileNotFoundError as err:
            print(f"Experiment File not found: {err}")


if __name__ == "__main__":

    DATE = 20220713
    WHEEL_WEIGHT = 50  # In N
    WHEEL_DIAMETER = 0.18  # In m
    WHEEL_RADIUS = WHEEL_DIAMETER / 2  # In m
    SLIP_CONSTANT = 1  # constant for angular velocity (should not matter)

    parser = argparse.ArgumentParser(
        description="Plotting the data for slip values"
    )
    parser.add_argument(
        "SR",
        nargs="+",
        help="Slip Ratio values",
        choices=["10", "30", "50", "70", "90"],
    )
    SR = parser.parse_args()
    SR = SR.SR  # list of slip ratio

    FIG, AX = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

    main()
    plt.show()
