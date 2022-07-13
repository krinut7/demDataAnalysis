"""PLot data collected from simulation and experiments."""

import pandas as pd
from matplotlib import pyplot as plt
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# import numpy as np

# from os.path import exists as file_exists

DATE = 20220713
WHEEL_WEIGHT = 50  # In N
WHEEL_DIAMETER = 0.18  # In m
WHEEL_RADIUS = WHEEL_DIAMETER / 2  # In m
SLIP_CONSTANT = 1  # constant for angular velocity (should not matter)
SR = list()  # slip ration value in percentage


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

    df = df.drop(df[df.Time < 1].index)
    df = df.drop(df[df.Time > df.iloc[-1]["Time"] - 2].index).reset_index()
    df["Time"] = df["Time"] - df.loc[0, "Time"]

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

    df = df.rename(columns={"time": "Time", ".wheel_sinkage": "Sinkage"})

    df = df.drop(df[df.Time < 1].index)
    df = df.drop(df[df.Time > df.iloc[-1]["Time"] - 2].index).reset_index()
    df["Time"] = df["Time"] - df.loc[0, "Time"]

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
    df["Slip"] = 1 - df["Omega_motor"] / df["Omega_conveying"]
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

    df = df.drop(df[df.Time < 2].index).reset_index()

    df["Fx/Fz"] = df["wheel.fx"] / WHEEL_WEIGHT
    df["Fx/Fz"] = df["Fx/Fz"].fillna(0)
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    df = df.rename(
        columns={"wheel.fx": "Fx", "wheel.fy": "Fy", "wheel.fz": "Fz"}
    )

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

    # print(df)
    return df


def force_slip(df: list) -> pd.DataFrame:
    """Calculate the average of force values for each slip."""
    df_force_avg = {"Slip": SR, "Force_Avg": list()}

    for df in df:
        df_force_avg["Force_Avg"].append(abs(df["Fx"]).mean())

    # print(df_sim_avg)
    return pd.DataFrame(df_force_avg)


def curve_fitting(df_fslip: dict):
    """Fit the curve using polynomial regression."""
    # pol_reg.predict(poly_reg.fit_transform(X))
    pol_reg_dict = dict()

    for key in df_fslip:
        df = df_fslip[key]
        X = df["Slip"].astype(int).to_numpy()
        y = df["Force_Avg"].to_numpy()
        print(X, y)
        poly_reg = PolynomialFeatures(degree=3)
        X_poly = poly_reg.fit_transform(X.reshape(-1, 1))
        print(X_poly)
        pol_reg = LinearRegression()
        pol_reg.fit(X_poly, y)
        pol_reg_dict[key] = pol_reg.predict(poly_reg.fit_transform(X))

    return pol_reg_dict


def plot_data(
    sr_len: int,
    df_exp: dict = None,
    df_sim: dict = None,
    df_fslip: dict = None,
    pol_reg_dict: dict = None,
):
    """Plot the data.

    Arguments:
        sr_len: length of the slip ratio list
        df_exp: dict for experiment with the dataframes for each slip ratio
        df_sim: dict for simulation with the dataframes for each slip ratio
        df_slip: dict for slip values of the experiment
    """
    if df_exp is None:
        df_exp = dict()
    if df_sim is None:
        df_sim = dict()
    if df_fslip is None:
        df_fslip = dict()
    if pol_reg_dict is None:
        pol_reg_dict = dict()

    fig = {"force": None, "sinkage": None, "slip": None, "fslip": None}
    ax = {"force": None, "sinkage": None, "slip": None, "fslip": None}

    plot_value = "Fx/Fz"

    fig["force"], ax["force"] = plt.subplots(constrained_layout=True)
    fig["sinkage"], ax["sinkage"] = plt.subplots(constrained_layout=True)
    fig["slip"], ax["slip"] = plt.subplots(constrained_layout=True)
    fig["fslip"], ax["fslip"] = plt.subplots(constrained_layout=True)

    ax["force"].set(
        xlabel="Time",
        ylabel=plot_value,
        title=f"{plot_value}: SR{SR}",
        autoscale_on=True,
    )
    ax["sinkage"].set(
        xlabel="Time",
        ylabel="Sinkage",
        title=f"Sinkage: SR{SR}",
        autoscale_on=True,
    )
    ax["slip"].set(
        xlabel="Time",
        ylabel="Slip",
        title=f"Slip: SR{SR}",
        autoscale_on=True,
    )
    ax["fslip"].set(
        xlabel="Slip",
        ylabel=plot_value,
        title=f"{plot_value} vs. Slip",
        autoscale_on=True,
    )

    for i in range(sr_len - 1):

        """ax["force"].plot(
            "Time",
            plot_value,
            data=df_exp["force"][i],
            linestyle="-",
            label=f"Exp SR{SR[i]}",
        )
        ax["sinkage"].plot(
            "Time",
            "Sinkage",
            data=df_exp["sinkage"][i],
            linestyle="-",
            label=f"Exp SR{SR[i]}",
        )
        ax["slip"].plot(
            "Time",
            "Slip",
            data=df_exp["slip"][i],
            linestyle="-",
            label=f"Exp SR{SR[i]}",
        )"""

        ax["force"].plot(
            "Time",
            plot_value,
            data=df_sim["force"][i],
            linestyle="solid",
            label=f"Sim SR{SR[i]}",
        )

        ax["sinkage"].plot(
            "Time",
            "Sinkage",
            data=df_sim["sinkage"][i],
            linestyle="-",
            label=f"Sim SR{SR[i]}",
        )
        ax["slip"].plot(
            "Time",
            "Slip",
            data=df_sim["slip"][i],
            linestyle="-",
            label=f"Sim SR{SR[i]}",
        )

    """print(df_fslip)
    for key in df_fslip:
        df = df_fslip[key]
        X = df["Slip"].to_numpy()
        ax["fslip"].scatter(
            "Slip",
            "Force_Avg",
            data=df,
            linestyle="-",
            label=f"{key} vs. Slip",
        )
        ax["fslip"].plot(X, pol_reg_dict[key])"""

    ax["force"].legend()
    ax["sinkage"].legend()
    ax["slip"].legend()
    ax["fslip"].legend()

    # ax["slip"].imshow()

    plt.show()


def main():
    """Call Main function.

    df_exp: dict for experiment with the dataframes for each slip ratio
    df_sim: dict for simulation with the dataframes for each slip ratio
    """
    df_exp = {"force": list(), "sinkage": list(), "slip": list()}
    df_sim = {"force": list(), "sinkage": list(), "slip": list()}
    df_fslip = dict()
    pol_reg_dict = dict()

    for i in range(len(sys.argv) - 1):
        try:
            df_exp["force"].append(exp_force(i))
            df_exp["sinkage"].append(exp_sinkage(i))
            df_exp["slip"].append(exp_slip(i))
        except FileNotFoundError as err:
            print(f"Experiment File not found: {err}")

        try:
            df_sim["force"].append(sim_force(i))
            df_sim["sinkage"].append(sim_sinkage(i))
            df_sim["slip"].append(sim_slip(i))
        except FileNotFoundError as err:
            print(f"Simulation File not found: {err}")

    df_fslip["exp"] = force_slip(df_exp["force"])
    df_fslip["sim"] = force_slip(df_sim["force"])

    # pol_reg_dict = curve_fitting(df_fslip)
    plot_data(len(sys.argv), df_exp, df_sim, df_fslip, pol_reg_dict)


if __name__ == "__main__":

    for i in range(len(sys.argv) - 1):  # getting a list of slip ratio values
        SR.append(sys.argv[i + 1])

    main()
