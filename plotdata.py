"""Plot data collected from simulation and experiments."""

from pyclbr import Function
from turtle import shape
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import numpy as np


def exp_force(i: int, j: int) -> pd.DataFrame:
    """Read the experiment force data."""
    data_type = "experimentData"
    csv_onwheel = "leptrino_force_torque_on_wheel-force_torque.csv"
    csv_inside = "leptrino_force_torque_center-force_torque.csv"

    filename_onwheel = f"../data/{data_type}/{SR[i]}/{RUN[j]}/{csv_onwheel}"
    filename_inside = f"../data/{data_type}/{SR[i]}/{RUN[j]}/{csv_inside}"

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

    df["Time"] = df_onwheel["Time"]
    df["Fx"] = df_onwheel["Fy"] - t1
    df["Fz"] = df_onwheel["Fz"] - t2

    df["Time"] = pd.to_datetime(df["Time"])
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    for x in df.index:
        df.loc[x, "Time"] = (
            df.loc[x, "Time"].seconds
            + df.loc[x, "Time"].microseconds / 1000000
        )

    df = df.drop(df[df.Time > df.iloc[-1]["Time"] - 2].index)
    df["Fz_caliberated"] = df["Fz"]
    # df.loc[df["Fz"] < 40, "Fz_caliberated"] = WHEEL_WEIGHT
    # df.loc[df["Fz"] > 60, "Fz_caliberated"] = WHEEL_WEIGHT
    df["Fx/Fz"] = abs(df["Fx"] / df["Fz"])
    df = df.drop(df[df["Fx/Fz"] > 1].index).reset_index()
    df["Moving_Avg_Fx"] = df["Fx"].rolling(SPAN).mean()
    df["Moving_Avg_Fz"] = df["Fz"].rolling(SPAN).mean()
    df["Moving_Avg_FxFz"] = df["Fx/Fz"].rolling(SPAN).mean()
    df = df.dropna().reset_index()
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    return df


def exp_sinkage(i: int, j: int) -> pd.DataFrame:
    """Read the experiment sinkage data."""
    data_type = "experimentData"
    csv_filename = "swt_driver-vertical_unit_log.csv"
    filename = f"../data/{data_type}/{SR[i]}/{RUN[j]}/{csv_filename}"

    df = pd.read_csv(filename, usecols=["time", ".wheel_sinkage"])
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
    df["Sinkage"] = (df["Sinkage"] - df.loc[0, "Sinkage"]) / 1000
    df["Sample"] = range(0, len(df))
    df["Moving_Avg_Sinkage"] = df["Sinkage"].rolling(SPAN).mean()
    df = df.dropna().reset_index()
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    return df


def exp_slip(i: int, j: int) -> pd.DataFrame:
    """Calculate slip for experiment."""
    data_type = "experimentData"
    csv_angular = "swt_driver-vertical_unit_log.csv"
    csv_trans = "swt_driver-longitudinal_unit_log.csv"

    filename_angular = f"../data/{data_type}/{SR[i]}/{RUN[j]}/{csv_angular}"
    filename_trans = f"../data/{data_type}/{SR[i]}/{RUN[j]}/{csv_trans}"

    df_angular = pd.read_csv(
        filename_angular, usecols=["time", ".wheel_motor_angular_vel"]
    )
    df_trans = pd.read_csv(
        filename_trans, usecols=["time", ".conveying_motor_angular_vel"]
    )

    df_angular.columns = ["Time", "omega"]
    df_trans.columns = ["Time", "omega"]
    df_angular["r_omega"] = 0.0
    df_trans["r_omega"] = 0.0

    df_trans["Time"] = pd.to_datetime(df_trans["Time"])
    df_angular["Time"] = pd.to_datetime(df_angular["Time"])
    df_angular["Time"] = df_angular["Time"] - df_angular.loc[0, "Time"]
    df_trans["Time"] = df_trans["Time"] - df_trans.loc[0, "Time"]

    for x in df_angular.index:
        df_angular.loc[x, "Time"] = (
            df_angular.loc[x, "Time"].seconds
            + df_angular.loc[x, "Time"].microseconds / 1000000
        )
    for x in df_trans.index:
        df_trans.loc[x, "Time"] = (
            df_trans.loc[x, "Time"].seconds
            + df_trans.loc[x, "Time"].microseconds / 1000000
        )

    for x in df_angular.index:
        if x + 1 > len(df_angular.index) - 1:
            break
        df_angular.loc[x, "r_omega"] = (
            df_angular.loc[x, "omega"] * WHEEL_DIAMETER
        )

    for x in df_trans.index:
        if x + 1 > len(df_trans.index) - 1:
            break
        df_trans.loc[x, "r_omega"] = (
            df_trans.loc[x, "omega"] * CONVEYING_RADIUS
        )

    df_angular = df_angular.drop(df_angular[df_angular.Time < 2].index)
    df_angular = df_angular.drop(
        df_angular[df_angular.Time > df_angular.iloc[-1]["Time"] - 2].index
    ).reset_index()

    df_trans = df_trans.drop(df_trans[df_trans.Time < 2].index).reset_index()

    df = pd.DataFrame()
    df["Time"] = df_angular["Time"]
    df["Slip"] = abs(1 - (df_trans["r_omega"] / df_angular["r_omega"])) * 100

    return df


def sim_force(i: int, j: int) -> pd.DataFrame:
    """Read the simulation force data."""
    data_type = "simulationData"
    csv_filename = "result_monitor.csv"
    filename = f"../data/{data_type}/{SR[i]}/{csv_filename}"

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
    df["Slip"] = int(SR[i])
    df["Moving_Avg_Fx"] = df["Fx"].rolling(SPAN).mean()
    df["Moving_Avg_FxFz"] = df["Fx/Fz"].rolling(SPAN).mean()

    return df


def sim_sinkage(i: int, j: int) -> pd.DataFrame:
    """Read the simulation sinkae data."""
    data_type = "simulationData"
    csv_filename = "CenterOfMass.txt"
    filename = f"../data/{data_type}/{SR[i]}/{csv_filename}"

    df = pd.read_csv(
        filename, sep=" ", names=["Time", "Sinkage"], skiprows=[0]
    )

    df = df.drop(df[df.Time < 2].index).reset_index()
    df["Time"] = df["Time"] - df.loc[0, "Time"]
    df["Sinkage"] = -1 * (df["Sinkage"] - df.loc[0, "Sinkage"]) * 1000
    df["Sample"] = range(0, len(df))
    df["Moving_Avg_Sinkage"] = df["Sinkage"].rolling(SPAN).mean()

    return df


def runs_avg(i: int, func: Function, func2=None):
    """Calculate the average of all runs."""
    df_list = list()

    for j in range(len(RUN)):
        df_list.append(func(i, j))
        if func2 is not None:
            df_list.append(func2(i, j))

    foo = pd.concat(df_list)
    df = foo.groupby(foo.index).mean()

    df["Time"] = 0
    EXP_TIME = 30
    dt = EXP_TIME / len(df.index)
    for x in df.index:
        if x + 1 > len(df.index) - 1:
            break
        df.loc[x + 1, "Time"] = df.loc[x, "Time"] + dt
    df["Sample"] = range(0, len(df))

    return df


def curve_fitting(df_list: list):
    """Fit the curve."""
    df = pd.concat(df_list)
    df = df.dropna().reset_index()
    # df = df.drop(df[df["Fx/Fz"] > 1].index).reset_index()
    model = np.poly1d(np.polyfit(df["Slip"], df["Fx/Fz"], DEGREE))

    return model


def main():
    """Call main function."""
    df_list_exp = list()
    df_list_sim = list()
    fig, ax = plt.subplots(constrained_layout=True)
    for i in range(len(SR)):
        ax.set(
            xlabel=r"Time (s)",
            ylabel=r"Sinkage (mm)",
        )
        try:
            # for j in range(len(RUN)):
            df_exp = runs_avg(i, exp_sinkage, exp_slip)
            df_list_exp.append(df_exp)
            ax.plot(
                "Time",
                "Moving_Avg_Sinkage",
                "-",
                label=rf"$i$: {SR[i]}",
                data=df_exp,
            )
        except FileNotFoundError as err:
            print(err)

        """try:
            df_sim = runs_avg(i, sim_force)
            df_list_sim.append(df_sim)
            ax.plot(
                "Time",
                "Moving_Avg_FxFz",
                "-",
                label=rf"$i$: {SR[i]}",
                data=df_sim,
            )
        except FileNotFoundError as err:
            print(err)"""

    """try:
        exp_model = curve_fitting(df_list_exp)
        ax.plot(
            POLYLINE,
            exp_model(POLYLINE),
            "-b",
            label="Experiment",
        )
    except:
        print("Experiment File not found")

    try:
        sim_model = curve_fitting(df_list_sim)
        ax.plot(
            POLYLINE,
            sim_model(POLYLINE),
            "-r",
            label="Simulation",
        )
    except:
        print("Simulation file not found")"""

    fig.set_figheight(7)
    fig.set_figwidth(12)
    # ax.set_ylim(0)
    ax.set_xlim(0, 25)
    # ax.set_xticks([0, 10, 30, 50, 70, 90])
    ax.legend(fontsize=15)
    plt.show()
    fig.savefig(
        "../figures/experiment/sinkage.pdf",
        bbox_inches="tight",
        format="pdf",
    )


if __name__ == "__main__":

    WHEEL_WEIGHT = 50  # In N
    WHEEL_DIAMETER = 0.18  # In m
    WHEEL_RADIUS = WHEEL_DIAMETER / 2  # In m
    CONVEYING_RADIUS = 0.0627  # In mm
    SPAN = 100
    ALPHA = 0.1
    POLYLINE = np.linspace(0, 95, 500)
    DEGREE = 3

    parser = argparse.ArgumentParser(
        description="Plotting the data for slip values"
    )
    parser.add_argument(
        "SR",
        nargs="+",
        help="Slip Ratio values",
    )
    parser.add_argument("--runs", nargs="+", help="Run value", type=int)
    arguments_ = parser.parse_args()
    SR = arguments_.SR
    RUN = arguments_.runs

    plt.rc("axes", labelsize=20)
    plt.rc("axes", titlesize=20)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)

    main()
