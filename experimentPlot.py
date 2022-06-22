import pandas as pd
import matplotlib.pyplot as plt

class manipulate_data:
    """Class to read, clean, and extract only required data
    """

    def __init__(self, date, sr, flag=None):
        """flag is used to define the type of data: simulation or experiment and quantity: force or sinkage
        """
        self._flag = flag
        self._date = date
        self._sr = sr
    
    @property
    def _read_data(self):
        """creates the filename based on the flags and then reads the data into df dataFrame
        """

        if self._flag['type'] == 'experiment':
            data_type = 'experimentData'
            
            if self._flag['quantity'] == 'force':
                data_quantity = 'leptrino_force_torque_on_wheel-force_torque.csv'
                filename = f"../data/{self._date}/{data_type}/{self._date}_{self._sr}/{data_quantity}"
                self._df = pd.read_csv(filename, usecols=['time', '.wrench.force.x', '.wrench.force.z'])

            elif self._flag['quantity'] == 'sinkage':
                data_quantity = 'swt_driver-vertical_unit_log.csv'
                filename = f"../data/{self._date}/{data_type}/{self._date}_{self._sr}/{data_quantity}"
                self._df = pd.read_csv(filename, usecols=['time', '.wheel_sinkage'])
                
            
        elif self._flag['type'] == 'simulation':
            data_type = 'simulationData'
            
            if self._flag['quantity'] == 'force':
                data_quantity = 'result_monitor.csv'
                filename = f"../data/{self._date}/{data_type}/{self._date}_{self._sr}/{data_quantity}"
                self._df = pd.read_csv(filename, header=1, usecols=['Time', 'wheel.fx', 'wheel.fz'])

            elif self._flag['quantity'] == 'sinkage':
                data_quantity = 'CenterOfMass.txt'
                filename = f"../data/{self._date}/{data_type}/{self._date}_{self._sr}/{data_quantity}"
                self._df = pd.read_csv(filename, sep=' ', names=['Time', 'Sinkage'], skiprows=[0])
            
        #print(self._df)
        
    @property
    def clean_data(self):
        """extract the data from the csv based on the flag values
        """
        self._read_data
        if self._flag['type'] == 'experiment':
            if self._flag['quantity'] == 'force':
                self._df['fx/fz'] = self._df['.wrench.force.x']/self._df['.wrench.force.z']
                self._df['time'] = pd.to_datetime(self._df['time'])
                self._df['time'] = self._df['time'] - self._df.loc[0, 'time']
                for x in self._df.index:
                    self._df.loc[x, 'time'] = self._df.loc[x, 'time'].seconds + self._df.loc[x, 'time'].microseconds/1000000
                self._df.rename(columns={'time': 'Time'}, inplace=True)
                #self._df = self._df.groupby(['time']).mean().reset_index()
            
            elif self._flag['quantity'] == 'sinkage':
                self._df['time'] = pd.to_datetime(self._df['time'])
                self._df['time'] = self._df['time'] - self._df.loc[0, 'time']
                for x in self._df.index:
                    self._df.loc[x, 'time'] = self._df.loc[x, 'time'].seconds + self._df.loc[x, 'time'].microseconds/1000000
                self._df.rename(columns={'time': 'Time'}, inplace=True)
                #print(self._df)
        
        
        elif self._flag['type'] == 'simulation':
            if self._flag['quantity'] == 'force':
                self._df['fx/fz'] = self._df["wheel.fx"]/self._df["wheel.fz"]
                self._df['fx/fz'].fillna(0, inplace=True)
                self._df['Time'] = self._df['Time'] - self._df.loc[0, 'Time']
                for x in self._df.index:
                    if not -20 <= self._df.loc[x, 'fx/fz'] <=20:
                        self._df.drop(x, inplace=True)
            
            elif self._flag['quantity'] == 'sinkage':
                self._df['Time'] = self._df['Time'] - self._df.loc[0, 'Time']
                
        return self._df

class plot_data:
    """Class to plot the data
    """




if __name__ == '__main__':
    
    date, sr = 20220615, 10

    data_values = {'type': 'experiment', 'quantity': 'sinkage'}
    data = manipulate_data(date, sr, data_values)
    dataFrame1 = data.clean_data
    #data.plot_data()
    print(dataFrame1)
    #print(type(dataFrame1.loc[0, 'time']))