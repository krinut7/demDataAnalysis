"""PLot data collected from simulation and experiments."""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import argparse


def exp_force(i: int, j: int) -> pd.DataFrame:
    """Read the experiment force data.

    Arguments:
        i (int): INdex for SR list
        j (int): Index for run list
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
        f"../data/{DATE}/{data_type}/"
        f"{DATE}_{SR[i]}/{RUN[j]}/{csv_filename_onwheel}"
    )
    filename_inside = (
        f"../data/{DATE}/{data_type}/"
        f"{DATE}_{SR[i]}/{RUN[j]}/{csv_filename_inside}"
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

    df["Time"] = pd.to_datetime(df["Time"])
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    for x in df.index:
        df.loc[x, "Time"] = (
            df.loc[x, "Time"].seconds
            + df.loc[x, "Time"].microseconds / 1000000
        )

    # df = df.drop(df[df.Time < 15].index).reset_index()
    # df = df.drop(df[df.Time > df.iloc[-1]["Time"] - 2].index).reset_index()
    df["Time"] = df["Time"] - df.loc[0, "Time"]
    df["Fz_caliberated"] = df["Fz"]
    df.loc[df["Fz"] < 40, "Fz_caliberated"] = WHEEL_WEIGHT
    df.loc[df["Fz"] > 60, "Fz_caliberated"] = WHEEL_WEIGHT
    df["Fx/Fz"] = df["Fx"] / df["Fz_caliberated"]
    df["Moving_Avg_Fx"] = df["Fx"].rolling(SPAN).mean()
    df["Moving_Avg_Fz"] = df["Fz_caliberated"].rolling(SPAN).mean()
    df["Moving_Avg_FxFz"] = df["Fx/Fz"].rolling(SPAN).mean()

    return df


def runs_force_avg(i: int):
    """Calculate the average of all runs."""
    df_force = list()
    df = pd.DataFrame()

    for j in range(len(RUN)):
        df_force.append(exp_force(i, j))

    df["Time"] = (
        df_force[0]["Time"] + df_force[1]["Time"] + df_force[2]["Time"]
    ) / 3
    df["Fx"] = abs(
        (df_force[0]["Fx"] + df_force[1]["Fx"] + df_force[2]["Fx"]) / 3
    )
    df["Fz"] = (
        df_force[0]["Fz_caliberated"]
        + df_force[1]["Fz_caliberated"]
        + df_force[2]["Fz_caliberated"]
    ) / 3

    df["Fx/Fz"] = df["Fx"] / df["Fz"]
    df["Time"] = df["Time"] - df.loc[0, "Time"]
    df["Sample"] = range(0, len(df))
    df["Moving_Avg_Fx"] = df["Fx"].rolling(SPAN).mean()
    df["Moving_Avg_Fz"] = df["Fz"].rolling(SPAN).mean()
    df["Moving_Avg_FxFz"] = df["Fx/Fz"].rolling(SPAN).mean()

    return df


def exp_sinkage(i: int, j: int) -> pd.DataFrame:
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
    filename = (
        f"../data/{DATE}/{data_type}/{DATE}_{SR[i]}/{RUN[j]}/{csv_filename}"
    )

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
    # df["Time"] = df["Time"] - df.loc[0, "Time"]

    # df.Sinkage = df.Sinkage / 1000
    df["Sinkage"] = -1 * (df["Sinkage"] - df.loc[0, "Sinkage"]) / 1000
    df["Sample"] = range(0, len(df))
    df["Moving_Avg_Sinkage"] = df["Sinkage"].rolling(SPAN).mean()

    return df


def runs_sinkage_avg(i: int):
    """Calculate the average of all runs."""
    df_sinkage = list()
    df = pd.DataFrame()

    for j in range(len(RUN)):
        df_sinkage.append(exp_sinkage(i, j))

    df["Time"] = (
        df_sinkage[0]["Time"] + df_sinkage[1]["Time"] + df_sinkage[2]["Time"]
    ) / 3
    df["Sinkage"] = (
        df_sinkage[0]["Sinkage"]
        + df_sinkage[1]["Sinkage"]
        + df_sinkage[2]["Sinkage"]
    ) / 3

    df["Sinkage"] = -1 * (df["Sinkage"] - df.loc[0, "Sinkage"])
    df["Sample"] = range(0, len(df))
    df["Moving_Avg_Sinkage"] = df["Sinkage"].rolling(SPAN).mean()

    return df


def exp_slip(i: int, j: int) -> pd.DataFrame:
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
        f"../data/{DATE}/{data_type}/"
        f"{DATE}_{SR[i]}/{RUN[j]}/{csv_filename_angular}"
    )
    filename_conveying = (
        f"../data/{DATE}/{data_type}/"
        f"{DATE}_{SR[i]}/{RUN[j]}/{csv_filename_trans}"
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
    df = df.drop(df[df.Time > 40].index).reset_index()
    df["Time"] = df["Time"] - df.loc[0, "Time"]
    df["Sample"] = range(0, len(df))
    df["Moving_Avg_Slip"] = df["Slip"].rolling(100).mean()

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
    df["Moving_Avg_Fx"] = df["Fx"].rolling(SPAN).mean()
    df["Moving_Avg_FxFz"] = df["Fx/Fz"].rolling(SPAN).mean()
    return df


def sim_torque(i: int) -> pd.DataFrame:
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
        usecols=["Time", "wheel.tx", "wheel.ty", "wheel.tz", "wheel.fz"],
    )
    df.columns = ["Time", "Tx", "Ty", "Tz", "Fz"]

    df = df.drop(df[df.Time < 2].index).reset_index()

    df["Fz"] = df["Fz"] + WHEEL_WEIGHT
    df["Ty/Fz"] = df["Ty"] / (df["Fz"] * WHEEL_RADIUS)
    df["Time"] = df["Time"] - df.loc[0, "Time"]

    df["Sample"] = range(0, len(df))
    # df["Moving_Avg_Torque"] = df[PLOT_TORQUE].rolling(SPAN).mean()
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
    df["Sinkage"] = -1 * (df["Sinkage"] - df.loc[0, "Sinkage"]) * 1000
    df["Sample"] = range(0, len(df))
    df["Moving_Avg_Sinkage"] = df["Sinkage"].rolling(SPAN).mean()

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
    df["Moving_Avg_Slip"] = df["Slip"].rolling(SPAN).mean()

    return df


def force_slip(df: list) -> pd.DataFrame:
    """Calculate the average of force values for each slip."""
    df_avg = {
        "Slip": SR,
        "Force_Avg_Fx": list(),
        "Force_Avg_Fz": list(),
        "Force_Avg_FxFz": list(),
    }

    for df in df:
        df_avg["Force_Avg_Fx"].append(abs(df["Fx"]).mean())
        df_avg["Force_Avg_Fz"].append(abs(df["Fz"]).mean())
        df_avg["Force_Avg_FxFz"].append(abs(df["Fx/Fz"]).mean())

    df_avg = pd.DataFrame(data=df_avg)
    df_avg.Slip = df_avg.Slip.astype(int)
    return df_avg


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
    i: int,
    df: pd.DataFrame,
    plot_grid: tuple,
    xaxis_: str,
    yaxis_: str,
    ylabel_: str,
    units_: str,
    title_: str,
    label_: str = " ",
    alpha_: float = 1.0,
):
    """Plot the data."""
    AX[plot_grid].set(
        xlabel="Time (s)",
        ylabel=f"{ylabel_} {units_}",
        title=f"{title_}",
    )
    AX[plot_grid].plot(
        xaxis_,
        yaxis_,
        data=df,
        linestyle="-",
        label=f"SR{SR[i]}",
        alpha=alpha_,
    )
    AX[plot_grid].set_xlim(0, df[xaxis_].max() + 0.5)
    AX[plot_grid].legend(ncol=5, loc="best", fontsize=8)


def main():
    """Call Main function.

    df_exp: dict for experiment with the dataframes for each slip ratio
    df_sim: dict for simulation with the dataframes for each slip ratio
    """
    df_sim = list()
    df_exp = list()

    for i in range(len(SR)):
        try:
            df_sim.append(sim_force(i))
        except FileNotFoundError as err:
            print(f"Simulation File not found: {err}")
        try:
            df_exp.append(runs_force_avg(i))
        except FileNotFoundError as err:
            print(f"Experiment File not found: {err}")

    # df_avg_sim = force_slip(df_sim)
    # df_avg_exp = force_slip(df_exp)

    for i in range(len(SR)):
        try:
            plot_data(
                i,
                runs_force_avg(i),
                (0, 0),
                "Time",
                "Moving_Avg_Fx",
                "Fx",
                "(N)",
                "Fx vs. Time",
                label_="Exp: Mvg. Avg",
            )
            """plot_data(
                i,
                runs_force_avg(i),
                (0, 0),
                "Time",
                "Fx",
                "Fx",
                "(N)",
                "Fx vs. Time",
                label_="Exp: Raw",
            )"""
            plot_data(
                i,
                runs_force_avg(i),
                (0, 1),
                "Time",
                "Moving_Avg_FxFz",
                "Fx/Fz",
                "(mm) ",
                "Fx/Fz vs. Time",
                label_="Exp: Mvg. Avg",
            )
            """plot_data(
                i,
                runs_force_avg(i),
                (0, 1),
                "Time",
                "Fx/Fz",
                "Fx/Fz",
                " ",
                "Fx/Fz vs. Time",
                label_="Exp: Raw",
                alpha_=ALPHA,
            )"""
            plot_data(
                i,
                runs_sinkage_avg(i),
                (1, 0),
                "Time",
                "Moving_Avg_Sinkage",
                "Sinkage",
                "(mm)",
                "Sinkage vs. Time",
                label_="Exp: Mvg. Avg",
            )
            """plot_data(
                i,
                runs_sinkage_avg(i),
                (1, 0),
                "Time",
                "Sinkage",
                "Sinkage",
                "(mm)",
                "Sinkage vs. Time",
                label_="Exp: Raw",
                alpha_=ALPHA,
            )"""
        except FileNotFoundError as err:
            print(f"Experiment File not found: {err}")

        """try:
            plot_data(
                i,
                sim_force(i),
                (0, 0),
                "Time",
                "Moving_Avg_Fx",
                "Fx",
                "(N)",
                "Fx vs. Time",
                label_="Sim: Mvg. Avg",
            )
            plot_data(
                i,
                sim_force(i),
                (0, 0),
                "Time",
                "Fx",
                "Fx",
                "(N)",
                "Fx vs. Time",
                label_="Sim: Raw",
                alpha_=ALPHA,
            )
            plot_data(
                i,
                sim_force(i),
                (0, 1),
                "Time",
                "Moving_Avg_FxFz",
                "Fx/Fz",
                " ",
                "Fx/Fz vs. Time",
                label_="Sim: Mvg. Avg",
            )
            plot_data(
                i,
                sim_force(i),
                (0, 1),
                "Time",
                "Fx/Fz",
                "Fx/Fz",
                " ",
                "Fx/Fz vs. Time",
                label_="Sim: Raw",
                alpha_=ALPHA,
            )
            plot_data(
                i,
                sim_sinkage(i),
                (1, 0),
                "Time",
                "Moving_Avg_Sinkage",
                "Sinkage",
                "(mm)",
                "Sinkage vs. Time",
                label_="Sim: Mvg. Avg",
            )
            plot_data(
                i,
                sim_sinkage(i),
                (1, 0),
                "Time",
                "Sinkage",
                "Sinkage",
                "(mm)",
                "Sinkage vs. Time",
                label_="Sim: Raw",
                alpha_=ALPHA,
            )

        except FileNotFoundError as err:
            print(f"Simulation File not found: {err}")"""

    """AX[1, 1].set(
        xlabel="Slip (%)",
        ylabel="Fx/Fz",
        title="Fx/Fz vs. Slip",
        autoscale_on=True,
    )
    AX[1, 1].plot(
        "Slip",
        "Force_Avg_FxFz",
        "^:",
        data=df_avg_exp,
        label="Experiment",
    )"""

    """AX[1, 1].plot(
        "Slip",
        "Force_Avg_FxFz",
        "^:",
        data=df_avg_sim,
        label="Simluation",
    )"""
    AX[1, 1].set_xlim(0, 100)
    AX[1, 1].set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    AX[1, 1].legend(ncol=2, loc="best", fontsize=6)


if __name__ == "__main__":

    DATE = 20220727
    WHEEL_WEIGHT = 50  # In N
    WHEEL_DIAMETER = 0.18  # In m
    WHEEL_RADIUS = WHEEL_DIAMETER / 2  # In m
    SLIP_CONSTANT = 1  # constant for angular velocity (should not matter)
    SPAN = 100
    SR = list()  # slip ration value in percentage
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
    parser.add_argument("--RUNS", nargs="+", help="Run value", type=int)
    arguments_ = parser.parse_args()
    SR = arguments_.SR
    RUN = arguments_.RUNS

    FIG, AX = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    FIG.set_figheight(7)
    FIG.set_figwidth(12)

    print("==== STARTING PLOTTING ====")
    main()
    # FIG.savefig(f"../figures/experiment/exp_moving_{SR}.png")
    plt.show()
    print("==== PLOTTING ENDED ====")
