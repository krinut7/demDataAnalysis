"""PLot data collected from simulation and experiments."""

import pandas as pd
import matplotlib.pyplot as plt
import sys

DATE = 20220615
SR = sys.argv[1]


def exp_force():
    """Read the experiment force data."""
    data_type = "experimentData"
    csv_filename = "leptrino_force_torque_on_wheel-force_torque.csv"
    filename = f"../data/{DATE}/{data_type}/{DATE}_{SR}/{csv_filename}"

    df = pd.read_csv(
        filename,
        usecols=[
            "time",
            ".wrench.force.x",
            ".wrench.force.y",
            ".wrench.force.z",
        ],
    )

    df["Fx/Fz"] = df[".wrench.force.x"] / df[".wrench.force.z"]
    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"] - df.loc[0, "time"]

    for x in df.index:
        df.loc[x, "time"] = (
            df.loc[x, "time"].seconds
            + df.loc[x, "time"].microseconds / 1000000
        )
    if not -1 <= df.loc[x, "Fx/Fz"] <= 1:  # removing extreme values
        df = df.drop(x)

    df = df.rename(
        columns={
            ".wrench.force.x": "Fx",
            ".wrench.force.y": "Fy",
            ".wrench.force.z": "Fz",
            "time": "Time",
        }
    )

    return df


def exp_sinkage():
    """Read the experiment sinkage data."""
    data_type = "experimentData"
    csv_filename = "swt_driver-vertical_unit_log.csv"
    filename = f"../data/{DATE}/{data_type}/{DATE}_{SR}/{csv_filename}"

    df = pd.read_csv(filename, usecols=["time", ".wheel_sinkage"])

    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"] - df.loc[0, "time"]
    df[".wheel_sinkage"] = df[".wheel_sinkage"] / 100000000
    for x in df.index:
        df.loc[x, "time"] = (
            df.loc[x, "time"].seconds
            + df.loc[x, "time"].microseconds / 1000000
        )

    _ = list(range(df.index.size - 5, df.index.size))
    df = df.drop(_)

    df = df.rename(columns={"time": "Time", ".wheel_sinkage": "Sinkage"})

    return df


def sim_force():
    """Read the simulation force data."""
    data_type = "simulationData"
    csv_filename = "result_monitor.csv"
    filename = f"../data/{DATE}/{data_type}/{DATE}_{SR}/{csv_filename}"

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

    print(x)
    # print(df)
    df["Fx/Fz"] = df["wheel.fx"] / df["wheel.fz"]
    # print(df)
    df["Fx/Fz"] = df["Fx/Fz"].fillna(0)
    # print(df)
    # df['Time'] = df['Time'] - df.loc[0, 'Time']

    # for x in df.index:
    # if not -20 <= df.loc[x, 'wheel.fz'] <= 20:  # removing extreme values
    # df = df.drop(x)

    df = df.rename(
        columns={"wheel.fx": "Fx", "wheel.fy": "Fy", "wheel.fz": "Fz"}
    )
    return df


def sim_sinkage():
    """Read the simulation sinkae data."""
    data_type = "simulationData"
    csv_filename = "CenterOfMass.txt"
    filename = f"../data/{DATE}/{data_type}/{DATE}_{SR}/{csv_filename}"

    df = pd.read_csv(
        filename, sep=" ", names=["Time", "Sinkage"], skiprows=[0]
    )

    df["Time"] = df["Time"] - df.loc[0, "Time"]
    df["Sinkage"] = df["Sinkage"] - df.loc[0, "Sinkage"]
    return df


def plot_data(df_exp=None, df_sim=None):
    """Plot the data."""
    if df_exp is None:
        df_exp = dict()
    if df_sim is None:
        df_sim = dict()

    plot_value = "Fx/Fz"
    fig, ax = plt.subplots(1, 2)

    ax[0].set(
        xlabel="Time",
        ylabel=plot_value,
        title=f"{plot_value}: SR{SR}",
        autoscale_on=True,
    )
    """ax[0].plot(
        'Time', 'Fx', data=df_exp['force'], linestyle='-',
        label='Experiment')"""
    ax[0].plot(
        "Time",
        plot_value,
        data=df_sim["force"],
        linestyle="-",
        label="Simulation",
    )

    ax[1].set(
        xlabel="Time",
        ylabel="Sinkage",
        title=f"Sinkage: SR{SR}",
        autoscale_on=True,
    )
    ax[1].plot(
        "Time",
        "Sinkage",
        data=df_exp["sinkage"],
        linestyle="-",
        label="Experiment",
    )
    """ax[1].plot(
        'Time', 'Sinkage', data=df_sim['sinkage'], linestyle='-',
        label='Simulation')"""

    ax[0].legend()
    ax[1].legend()
    plt.show()


def main():
    """Call Main function."""
    df_exp = dict()
    df_sim = dict()

    df_exp["force"] = exp_force()
    df_exp["sinkage"] = exp_sinkage()

    df_sim["force"] = sim_force()
    df_sim["sinkage"] = sim_sinkage()

    # print(SR)
    # print(f'Experiment:\n{df_exp}\nSimulation:\n{df_sim}')
    print(f"Force:\n{df_sim['force'].to_string()}")
    plot_data(df_exp=df_exp, df_sim=df_sim)


if __name__ == "__main__":
    main()
