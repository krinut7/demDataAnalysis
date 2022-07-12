"""PLot data collected from simulation and experiments."""

import pandas as pd
import matplotlib.pyplot as plt
import sys

# from os.path import exists as file_exists

DATE = 20220705
WHEEL_WEIGHT = 50  # In N
SLIP_CONSTANT = 1  # constant for angular velocity (should not matter)
SR = list()  # slip ration value in percentage


def exp_force(i: int) -> pd.DataFrame:
    """Read the experiment force data.

    The axis for on wheel and estimator are different.
    Estimator is the conventional axis.
    Estimator -> On wheel: Fx -> -Fy, Fy -> -Fx, Fz -> -Fz
                           Mx -> -My, My -> -Mz, Mz -> -Mx
    """
    data_type = "experimentData"
    csv_filename = "leptrino_force_torque_on_wheel-force_torque.csv"
    filename = f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{csv_filename}"

    df = pd.read_csv(
        filename,
        usecols=[
            "time",
            ".wrench.force.x",
            ".wrench.force.y",
            ".wrench.force.z",
        ],
    )

    df = df.rename(
        columns={
            ".wrench.force.x": "Fy",  # check out the docstring
            ".wrench.force.y": "Fx",
            ".wrench.force.z": "Fz",
            "time": "Time",
        }
    )

    # df["Fx"] = -1 * df["Fx"]
    df["Fy"] = -1 * df["Fy"]  # changing the sign
    df["Fz"] = -1 * df["Fz"]

    df["Fx/Fz"] = df["Fx"] / WHEEL_WEIGHT
    df["Time"] = pd.to_datetime(df["Time"])
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    for x in df.index:
        df.loc[x, "Time"] = (
            df.loc[x, "Time"].seconds
            + df.loc[x, "Time"].microseconds / 1000000
        )
    # if not -1 <= df.loc[x, "Fx/Fz"] <= 1:  # removing extreme values
    # df = df.drop(x)
    return df


def exp_sinkage(i: int) -> pd.DataFrame:
    """Read the experiment sinkage data."""
    data_type = "experimentData"
    csv_filename = "swt_driver-vertical_unit_log.csv"
    filename = f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{csv_filename}"

    df = pd.read_csv(filename, usecols=["time", ".wheel_sinkage"])

    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"] - df.loc[0, "time"]

    for x in df.index:
        df.loc[x, "time"] = (
            df.loc[x, "time"].seconds
            + df.loc[x, "time"].microseconds / 1000000
        )

    for x in df.index:
        if df.loc[x, ".wheel_sinkage"] != 0:
            df = df.drop(x)
        else:
            break

    df = df.reset_index()

    _ = list(range(df.index.size - 5, df.index.size))
    df = df.drop(_)

    df = df.rename(columns={"time": "Time", ".wheel_sinkage": "Sinkage"})
    # print(df)
    return df


def exp_slip(i: int) -> pd.DataFrame:
    """Read and calculate the slip values for experiment."""
    data_type = "experimentData"
    csv_filename_angular = "swt_driver-vertical_unit_log.csv"
    csv_filename_trans = "swt_driver-longitudinal_unit_log.csv"

    filename_angular = (
        f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{csv_filename_angular}"
    )
    filename_trans = (
        f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{csv_filename_trans}"
    )

    df_angular = pd.read_csv(
        filename_angular, usecols=["time", ".wheel_motor_angular_vel"]
    )
    df_trans = pd.read_csv(
        filename_trans, usecols=["time", ".conveying_motor_angular_vel"]
    )

    df_slip["Time"] = df_angular["time"]


def sim_force(i: int) -> pd.DataFrame:
    """Read the simulation force data."""
    data_type = "simulationData"
    csv_filename = "result_monitor.csv"
    filename = f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{csv_filename}"

    df = pd.read_csv(
        filename,
        header=1,
        usecols=["Time", "wheel.fx", "wheel.fy", "wheel.fz"],
    )

    for x in df.index:
        if df.loc[x, "Time"] < 2:
            df = df.drop(x)
        else:
            break

    df = df.reset_index()

    df["Fx/Fz"] = df["wheel.fx"] / WHEEL_WEIGHT
    df["Fx/Fz"] = df["Fx/Fz"].fillna(0)
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    df = df.rename(
        columns={"wheel.fx": "Fx", "wheel.fy": "Fy", "wheel.fz": "Fz"}
    )

    return df


def sim_sinkage(i: int) -> pd.DataFrame:
    """Read the simulation sinkae data."""
    data_type = "simulationData"
    csv_filename = "CenterOfMass.txt"
    filename = f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{csv_filename}"

    df = pd.read_csv(
        filename, sep=" ", names=["Time", "Sinkage"], skiprows=[0]
    )

    for x in df.index:
        if df.loc[x, "Time"] < 2:
            df = df.drop(x)
        else:
            break

    df = df.reset_index()
    df["Time"] = df["Time"] - df.loc[0, "Time"]
    df["Sinkage"] = df["Sinkage"] - df.loc[0, "Sinkage"]

    return df


def plot_data(sr_len: int, df_exp: dict = None, df_sim: dict = None):
    """Plot the data.

    Arguments:
        sr_len: length of the slip ratio list
        df_exp: dict for experiment with the dataframes for each slip ratio
        df_sim: dict for simulation with the dataframes for each slip ratio
    """
    if df_exp is None:
        df_exp = dict()
    if df_sim is None:
        df_sim = dict()

    plot_value = "Fx/Fz"
    fig, ax = plt.subplots(1, 2, constrained_layout=True)

    ax[0].set(
        xlabel="Time",
        ylabel=plot_value,
        title=f"{plot_value}: SR{SR}",
        autoscale_on=True,
    )
    ax[1].set(
        xlabel="Time",
        ylabel="Sinkage",
        title=f"Sinkage: SR{SR}",
        autoscale_on=True,
    )

    for i in range(sr_len - 1):

        """ax[0].plot(
            "Time",
            plot_value,
            data=df_exp["force"][i],
            linestyle="-",
            label=f"Exp SR{SR[i]}",
        )"""
        """ax[1].plot(
            "Time",
            "Sinkage",
            data=df_exp["sinkage"][i],
            linestyle="-",
            label=f"Exp SR{SR[i]}",
        )"""

        ax[0].plot(
            "Time",
            plot_value,
            data=df_sim["force"][i],
            linestyle="solid",
            label=f"Sim SR{SR[i]}",
        )

        ax[1].plot(
            "Time",
            "Sinkage",
            data=df_sim["sinkage"][i],
            linestyle="-",
            label=f"Sim SR{SR[i]}",
        )

    ax[0].legend()
    ax[1].legend()
    plt.show()


def main():
    """Call Main function.

    df_exp: dict for experiment with the dataframes for each slip ratio
    df_sim: dict for simulation with the dataframes for each slip ratio
    """
    df_exp = {"force": list(), "sinkage": list()}
    df_sim = {"force": list(), "sinkage": list()}

    for i in range(len(sys.argv) - 1):
        try:
            df_exp["force"].append(exp_force(i))
            df_exp["sinkage"].append(exp_sinkage(i))
        except FileNotFoundError as err:
            print(f"Experiment File not found: {err}")

        try:
            df_sim["force"].append(sim_force(i))
            df_sim["sinkage"].append(sim_sinkage(i))
        except FileNotFoundError as err:
            print(f"Simulation File not found: {err}")

    plot_data(len(sys.argv), df_exp, df_sim)


if __name__ == "__main__":

    for i in range(len(sys.argv) - 1):  # getting a list of slip ratio values
        SR.append(sys.argv[i + 1])

    main()
