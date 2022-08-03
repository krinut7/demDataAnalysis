"""Plot data collected from simulation and experiments."""

from pyclbr import Function
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import argparse


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

    df["Fx"] = df_onwheel["Fy"] - t1
    df["Fz"] = df_onwheel["Fz"] - t2
    df["Time"] = df_onwheel["Time"]

    df["Time"] = pd.to_datetime(df["Time"])
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    for x in df.index:
        df.loc[x, "Time"] = (
            df.loc[x, "Time"].seconds
            + df.loc[x, "Time"].microseconds / 1000000
        )

    df["Fz_caliberated"] = df["Fz"]
    # df.loc[df["Fz"] < 40, "Fz_caliberated"] = WHEEL_WEIGHT
    # df.loc[df["Fz"] > 60, "Fz_caliberated"] = WHEEL_WEIGHT
    df["Fx/Fz"] = df["Fx"] / df["Fz_caliberated"]
    df["Moving_Avg_Fx"] = df["Fx"].rolling(SPAN).mean()
    df["Moving_Avg_Fz"] = df["Fz_caliberated"].rolling(SPAN).mean()
    df["Moving_Avg_FxFz"] = df["Fx/Fz"].rolling(SPAN).mean()

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
    df["Sinkage"] = -1 * (df["Sinkage"] - df.loc[0, "Sinkage"]) / 1000
    df["Sample"] = range(0, len(df))
    df["Moving_Avg_Sinkage"] = df["Sinkage"].rolling(SPAN).mean()

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
        # dt = df_angular.loc[x + 1, "Time"] - df_angular.loc[x, "Time"]
        df_angular.loc[x, "r_omega"] = (
            df_angular.loc[x, "omega"] * WHEEL_DIAMETER
        )

    for x in df_trans.index:
        if x + 1 > len(df_trans.index) - 1:
            break
        # dt = df_trans.loc[x + 1, "Time"] - df_trans.loc[x, "Time"]
        df_trans.loc[x, "r_omega"] = (
            df_trans.loc[x, "omega"] * CONVEYING_RADIUS
        )

    df_angular = df_angular.drop(df_angular[df_angular.Time < 2].index)
    df_angular = df_angular.drop(
        df_angular[df_angular.Time > df_angular.iloc[-1]["Time"] - 2].index
    )

    df_trans = df_trans.drop(df_trans[df_trans.Time < 2].index)

    df = pd.DataFrame()
    df["Time"] = df_angular["Time"]
    df["Slip"] = 1 - (df_trans["r_omega"] / df_angular["r_omega"])
    print(df_angular)
    print(df_trans)
    print(df)

    return df


def sim_force(i: int) -> pd.DataFrame:
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


def sim_sinkage(i: int) -> pd.DataFrame:
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


def runs_avg(i: int, func: Function):
    """Calculate the average of all runs."""
    df_list = list()

    for j in range(len(RUN)):
        df_list.append(func(i, j))

    foo = pd.concat(df_list)
    df = foo.groupby(level=foo.index).mean()

    return df


def main():
    """Call main function."""
    fig, ax = plt.subplots(constrained_layout=True)
    for i in range(len(SR)):
        ax.set(
            xlabel="Time (s)",
            ylabel="Fz (N)",
            title=f"Fz vs. Time i:{SR[i]}",
        )
        for j in range(len(RUN)):
            ax.plot(
                "Time",
                "Moving_Avg_Fz",
                data=exp_force(i, j),
                linestyle="-",
                label=f"i:{SR[i]} run:{RUN[j]}",
            )
    # ax.set_ylim(-0.1, 1)
    # ax.set_yticks([0, 0.1, 0.3, 0.5, 0.7, 0.9])
    ax.legend()
    plt.show()


if __name__ == "__main__":

    WHEEL_WEIGHT = 50  # In N
    WHEEL_DIAMETER = 0.18  # In m
    WHEEL_RADIUS = WHEEL_DIAMETER / 2  # In m
    CONVEYING_RADIUS = 0.0627  # In mm
    SPAN = 100
    ALPHA = 0.1

    parser = argparse.ArgumentParser(
        description="Plotting the data for slip values"
    )
    parser.add_argument(
        "SR",
        nargs="+",
        help="Slip Ratio values",
        choices=["00", "10", "30", "50", "70", "90"],
    )
    parser.add_argument("--runs", nargs="+", help="Run value", type=int)
    arguments_ = parser.parse_args()
    SR = arguments_.SR
    RUN = arguments_.runs

    main()
