"""PLot data collected from simulation and experiments."""

from cmath import exp
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
        df_force = {'exp': None, 'sim': None}
        df_sinkage = {'exp': None, 'sim': None}

        if self._flag['type'] == 'experiment':
            data_type = 'experimentData'

            if self._flag['quantity'] == 'force':
                data_quantity = (
                    'leptrino_force_torque_on_wheel-force_torque.csv')
                filename = (
                    f"../data/{self._date}/{data_type}/"
                    f"{self._date}_{self._sr}/{data_quantity}")

                df_force['exp'] = pd.read_csv(
                    filename, usecols=[
                        'time', '.wrench.force.x', '.wrench.force.z'])

            elif self._flag['quantity'] == 'sinkage':
                data_quantity = 'swt_driver-vertical_unit_log.csv'

                filename = (
                    f"../data/{self._date}/{data_type}/"
                    f"{self._date}_{self._sr}/{data_quantity}")

                df_sinkage['exp'] = pd.read_csv(
                    filename, usecols=['time', '.wheel_sinkage'])

        elif self._flag['type'] == 'simulation':
            data_type = 'simulationData'

            if self._flag['quantity'] == 'force':
                data_quantity = 'result_monitor.csv'

                filename = (
                    f"../data/{self._date}/{data_type}/"
                    f"{self._date}_{self._sr}/{data_quantity}")

                df_force['sim'] = pd.read_csv(
                    filename, header=1, usecols=[
                        'Time', 'wheel.fx', 'wheel.fz'])

            elif self._flag['quantity'] == 'sinkage':
                data_quantity = 'CenterOfMass.txt'

                filename = (
                    f"../data/{self._date}/{data_type}/"
                    f"{self._date}_{self._sr}/{data_quantity}")

                df_sinkage['sim'] = pd.read_csv(
                    filename, sep=' ', names=['Time', 'Sinkage'],
                    skiprows=[0])

        return df_force, df_sinkage

    def clean_data(self, df_force, df_sinkage):
        """Extract the data from the csv based on the flag values.

        Create columns with correct name and datatype.

        Arguments:
            df_force (Pandas DataFrame): df_force with read data.

        Returns:
            df (Pandas DataFrame): extracted data with the correct column name
            and type.
        """
        if self._flag['type'] == 'experiment':
            if self._flag['quantity'] == 'force':
                df = df_force['exp']
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
                df_force['exp'] = df

            elif self._flag['quantity'] == 'sinkage':
                df = df_sinkage['exp']
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
                df_sinkage['exp'] = df

        elif self._flag['type'] == 'simulation':
            if self._flag['quantity'] == 'force':
                df = df_force['sim']
                df['Fx/Fz'] = df["wheel.fx"] / df["wheel.fz"]
                df = df['Fx/Fz'].fillna(0)
                df['Time'] = df['Time'] - df.loc[0, 'Time']
                for x in df.index:
                    if not -5 <= df.loc[x, 'Fx/Fz'] <= 5:
                        df = df.drop(x)
                df = df.rename(columns={'wheel.fx': 'Fx', 'wheel.fz': 'Fz'})
                df_force['sim'] = df

            elif self._flag['quantity'] == 'sinkage':
                df = df_sinkage['sim']
                df['Time'] = df['Time'] - df.loc[0, 'Time']
                df_sinkage['sim'] = df

        return df_force, df_sinkage

    def plot_data(self, df_force, df_sinkage):
        """Plot data for a single Slip ratio.

        Arguments:
            df (Pandas DataFrame): df with clean data.
        """
        fig, ax = plt.subplots(1, 2)

        if self._flag['quantity'] == 'force':
            ax[0, 0].set(
                xlabel='Time', ylabel='Fx/Fz', title=f'Fx/Fz: SR{self._sr}',
                autoscale_on=True)  # , xlim=(0, 45))
            ax[0, 0].plot(
                'Time', 'Fx/Fz', data=df_force['exp'], linestyle='-',
                label='Experiment')
            ax[0, 0].plot(
                'Time', 'Fx/Fz', data=df_force['exp'], linestyle='-',
                label='Simualation')
        elif self._flag['quantity'] == 'sinkage':
            ax[0, 1].set(
                xlabel='Time', ylabel='Sinkage',
                title=f'Sinkage: SR{self._sr}', autoscale_on=True)
            ax[0, 1].plot(
                'Time', 'Sinkage', data=df_sinkage[exp], linestyle='-',
                label='Experiment')
            ax[0, 1].plot(
                'Time', 'Sinkage', data=df_sinkage[exp], linestyle='-',
                label='Simulation')

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
