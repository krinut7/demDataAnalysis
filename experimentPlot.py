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
            elif self._flag['quantity'] == 'sinkage':
                data_quantity = 'swt_driver-vertical_unit_log'
            
            filename = f"../data/{self._date}/{data_type}/{self._date}_{self._sr}/{data_quantity}"
            self._df = pd.read_csv(filename)
        
        elif self._flag['type'] == 'simulation':
            data_type = 'simulationData'
            
            if self._flag['quantity'] == 'force':
                data_quantity = 'result_monitor.csv'
            elif self._flag['quantity'] == 'sinkage':
                data_quantity = 'CenterOfMass.txt'
            
            filename = f"../data/{self._date}/{data_type}/{self._date}_{self._sr}/{data_quantity}"
            self._df = pd.read_csv(filename, skiprows=[0])
        

    def _extract_data(self):
        """extract the data from the csv based on the flag values
        """

        if self._flag['type'] == 'experiment':
            self._df = self._df[["time", ".wrench.force.x", ".wrench.force.z"]]
            self._df['fx/fz'] = self._df[".wrench.force.x"]/self._df[".wrench.force.z"]
            self._df['time'] = pd.to_datetime(self._df['time'])
            self._df.rename(columns={'time': 'Time'}, inplace=True)
            #self._df['timeCalculated'] = self._df['Time'] - self._df.loc[0, 'Time']
            #for x in self._df.index:
                #self._df.loc[x, 'timeCalculated'] = self._df.loc[x, 'timeCalculated'].seconds
        
        elif self._flag['type'] == 'simulation':
            self._df = self._df[["Time", "wheel.fx", "wheel.fz"]]
            self._df['fx/fz'] = self._df["wheel.fx"]/self._df["wheel.fz"]
            self._df['fx/fz'].fillna(0, inplace=True)
            self._df['Time'] = self._df['Time'] - self._df.loc[0, 'Time']
            for x in self._df.index:
                if not -20 <= self._df.loc[x, 'fx/fz'] <=20:
                    self._df.drop(x, inplace=True)
        
        print(self._df)

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
    data1_values = {'type': 'experiment', 'quantity': 'force'}
    data1 = manipulate_data(date, sr, data1_values)
    data1.plot_data()
    

    data2_values = {'type': 'simulation', 'quantity': 'force'}
    data2 = manipulate_data(date, sr, data2_values)
    data2.plot_data()