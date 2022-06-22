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
        

    def _extract_data(self):
        """extract the data from the csv based on the flag values
        """
        self._read_data()
        if self._flag['type'] == 'experiment':
            if self._flag['quantity'] == 'force':
                self._df['fx/fz'] = self._df['.wrench.force.x']/self._df['.wrench.force.z']
                self._df['time'] = pd.to_datetime(self._df['time'])
                self._df.rename(columns={'time': 'Time'}, inplace=True)
                self._df['timeCalculated'] = self._df['Time'] - self._df.loc[0, 'Time']
                for x in self._df.index:
                    self._df.loc[x, 'timeCalculated'] = self._df.loc[x, 'timeCalculated'].microseconds
            
            elif self._flag['quantity'] == 'sinkage':
                self._df['time'] = pd.to_datetime(self._df['time'])
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
                
        print(self._df)
        print(self._df['Time'])

    def plot_data(self):
        """plotting the data
        """
        self._read_data()
        self._extract_data()
        #self._df.plot()
        plt.plot(self._df['Time'], self._df['fx/fz'], '-')
        plt.show()



if __name__ == '__main__':
    
    date, sr = 20220615, 10
    """
    data1_values = {'type': 'experiment', 'quantity': 'sinkage'}
    data1 = manipulate_data(date, sr, data1_values)
    data1._extract_data()

    data2_values = {'type': 'simulation', 'quantity': 'force'}
    data2 = manipulate_data(date, sr, data2_values)
    data2._read_data()
    """
    data_values = {'type': 'experiment', 'quantity': 'force'}
    data = manipulate_data(date, sr, data_values)
    data._read_data()