"""PLot data collected from simulation and experiments."""

import pandas as pd
import matplotlib.pyplot as plt
import sys


class DataFrame:
    """Class to read, clean, and extract only required data.

    Attributes:
        date (int): Date of data collection yyyymmdd
        sr (int): Slip ratio
        flag (dictionary): Type and quantity of data
    """

    def __init__(self, date, sr, flag=None):
        """Initialize DataFrame class.

        Paramters:
            date (int): Date of data collection yyyymmdd
            sr (int): Slip ratio
            flag (dictionary): Type: experiment or simulation
                                Quantity: force or sinkage
        """
        self._flag = flag
        self._date = date
        self._sr = sr

    def read_data(self):
        """Read data from csv and load to dataframe based on flagvalues.

        Returns:
            df (Pandas DataFrame): read dataframe
        """
        if self._flag['type'] == 'experiment':
            data_type = 'experimentData'

            if self._flag['quantity'] == 'force':
                data_quantity = (
                    'leptrino_force_torque_on_wheel-force_torque.csv')
                filename = (
                    f"../data/{self._date}/{data_type}/"
                    f"{self._date}_{self._sr}/{data_quantity}")

                df = pd.read_csv(
                    filename, usecols=[
                        'time', '.wrench.force.x', '.wrench.force.z'])

            elif self._flag['quantity'] == 'sinkage':
                data_quantity = 'swt_driver-vertical_unit_log.csv'

                filename = (
                    f"../data/{self._date}/{data_type}/"
                    f"{self._date}_{self._sr}/{data_quantity}")

                df = pd.read_csv(filename, usecols=['time', '.wheel_sinkage'])

        elif self._flag['type'] == 'simulation':
            data_type = 'simulationData'

            if self._flag['quantity'] == 'force':
                data_quantity = 'result_monitor.csv'
                filename = (
                    f"../data/{self._date}/{data_type}/"
                    f"{self._date}_{self._sr}/{data_quantity}")
                df = pd.read_csv(
                    filename, header=1, usecols=[
                        'Time', 'wheel.fx', 'wheel.fz'])

            elif self._flag['quantity'] == 'sinkage':
                data_quantity = 'CenterOfMass.txt'
                filename = (
                    f"../data/{self._date}/{data_type}/"
                    f"{self._date}_{self._sr}/{data_quantity}")
                df = pd.read_csv(
                    filename, sep=' ', names=['Time', 'Sinkage'],
                    skiprows=[0])

        return df

    def clean_data(self, df):
        """Extract the data from the csv based on the flag values.

        Create columns with correct name and datatype.

        Arguments:
            df (Pandas DataFrame): df with read data.

        Returns:
            df (Pandas DataFrame): extracted data with the correct column name
            and type.
        """
        if self._flag['type'] == 'experiment':
            if self._flag['quantity'] == 'force':
                df['Fx/Fz'] = df['.wrench.force.x'] / df['.wrench.force.z']
                df['time'] = pd.to_datetime(df['time'])
                df['time'] = df['time'] - df.loc[0, 'time']

                for x in df.index:
                    df.loc[x, 'time'] = (df.loc[x, 'time'].seconds
                                         + df.loc[x, 'time'].microseconds
                                         / 1000000)

                    if not -0.3 <= df.loc[x, 'Fx/Fz'] <= 0.3:
                        df = df.drop(x)

                df = df.rename(columns={
                    '.wrench.force.x': 'Fx', '.wrench.force.z': 'Fz',
                    'time': 'Time'})

            elif self._flag['quantity'] == 'sinkage':
                df['time'] = pd.to_datetime(df['time'])
                df['time'] = df['time'] - df.loc[0, 'time']
                for x in df.index:
                    df.loc[x, 'time'] = (df.loc[x, 'time'].seconds
                                         + df.loc[x, 'time'].microseconds
                                         / 1000000)

                _ = list(range(df.index.size - 5, df.index.size))
                df = df.drop(_)

                df = df.rename(columns={
                    'time': 'Time', '.wheel_sinkage': 'Sinkage'})

        elif self._flag['type'] == 'simulation':
            if self._flag['quantity'] == 'force':
                df['Fx/Fz'] = df["wheel.fx"] / df["wheel.fz"]
                df = df['Fx/Fz'].fillna(0)
                df['Time'] = df['Time'] - df.loc[0, 'Time']
                for x in df.index:
                    if not -5 <= df.loc[x, 'Fx/Fz'] <= 5:
                        df = df.drop(x)
                df = df.rename(columns={'wheel.fx': 'Fx', 'wheel.fz': 'Fz'})

            elif self._flag['quantity'] == 'sinkage':
                df['Time'] = df['Time'] - df.loc[0, 'Time']

        return df

    def plot_data(self, df):
        """Plot data for a single Slip ratio.

        Arguments:
            df (Pandas DataFrame): df with clean data.
        """
        fig, ax = plt.subplots()

        if self._flag['quantity'] == 'force':
            ax.set(
                xlabel='Time', ylabel='Fx/Fz', title='Fx/Fz',
                autoscale_on=True, xlim=(0, 45))
            plt.plot('Time', 'Fx/Fz', data=df, linestyle='-')
        elif self._flag['quantity'] == 'sinkage':
            ax.set(
                xlabel='Time', ylabel='Sinkage', title='Sinkage',
                autoscale_on=True, xlim=(0, 20))
            plt.plot('Time', 'Sinkage', data=df, linestyle='-')

        plt.show()

    def plot_data_compare(self, data=None):
        """Plot data for all the slip ratios.

        Arguments:
            data (list): dataFrame for all the slipratio
        """
        _ = 10
        fig, ax = plt.subplots()

        for i in range(0, 4):
            if self._flag['type'] == 'experiment':
                if self._flag['quantity'] == 'force':
                    ax.set(
                        xlabel='Time', ylabel='Fx/Fz',
                        title='Fx/Fz Experiment', autoscale_on=True,
                        xlim=(0, 20))
                    ax.plot(
                        'Time', 'Fx/Fz', data=data[i],
                        linestyle='-', label=f'SR{_}')
                elif self._flag['quantity'] == 'sinkage':
                    ax.set(
                        xlabel='Time', ylabel='Sinkage',
                        title='Sinkage Experiment', autoscale_on=True)
                    ax.plot(
                        'Time', 'Sinkage', data=data[i],
                        linestyle='-', label=f'SR{_}')

            elif self._flag['type'] == 'simulation':
                if self._flag['quantity'] == 'force':
                    ax.set(
                        xlabel='Time', ylabel='Fx/Fz',
                        title='Fx/Fz Simulation', autoscale_on=True,
                        xlim=(0, 2))
                    ax.plot(
                        'Time', 'Fx/Fz', data=data[i],
                        linestyle='-', label=f'SR{_}')
                elif self._flag['quantity'] == 'sinkage':
                    ax.set(
                        xlabel='Time', ylabel='Sinkage',
                        title='Sinkage Simulation', autoscale_on=True)
                    ax.plot(
                        'Time', 'Sinkage', data=data[i],
                        linestyle='-', label=f'SR{_}')
            _ = _ + 20
        plt.legend()
        plt.show()
        # return cls


if __name__ == '__main__':

    data_values = {'type': f'{sys.argv[1]}', 'quantity': f'{sys.argv[2]}'}
    date = 20220629
    SR = [10, 30, 50, 70]
    dataFrame_object = []
    read_dataFrame = []
    clean_dataFrame = []

    for i in range(0, 4):
        dataFrame_object.insert(i, DataFrame(date, SR[i], data_values))
        read_dataFrame.insert(i, dataFrame_object[i].read_data())
        clean_dataFrame.insert(
            i, dataFrame_object[i].clean_data(read_dataFrame[i]))
        # clean_dataFrame.insert(i, dataFrame_object[i].clean_data())

    if sys.argv[3] == 'single':
        print(f"Read DataFrame:\n{read_dataFrame[0]}")
        print(f"Clean DataFrame\n{clean_dataFrame[0]}")
        dataFrame_object[0].plot_data(clean_dataFrame[0])

    elif sys.argv[3] == 'compare':
        dataFrame_object[0].plot_data_compare(clean_dataFrame)

    # print(help(dataFrame_object[0].clean_data))
