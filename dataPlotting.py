import pandas as pd
import matplotlib.pyplot as plt
import sys

class data_frame:
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
        
    @property
    def clean_data(self):
        """extract the data from the csv based on the flag values
        """
        self._read_data
        if self._flag['type'] == 'experiment':
            if self._flag['quantity'] == 'force':
                self._df['Fx/Fz'] = self._df['.wrench.force.x']/self._df['.wrench.force.z']
                self._df['time'] = pd.to_datetime(self._df['time'])
                self._df['time'] = self._df['time'] - self._df.loc[0, 'time']
                for x in self._df.index:
                    self._df.loc[x, 'time'] = self._df.loc[x, 'time'].seconds + self._df.loc[x, 'time'].microseconds/1000000
                    if not -0.3 <= self._df.loc[x, 'Fx/Fz'] <=0.3:
                        self._df.drop(x, inplace=True)
                self._df.rename(columns={'.wrench.force.x': 'Fx', '.wrench.force.z': 'Fz', 'time': 'Time'}, inplace=True)
                #self._df = self._df.groupby(['time']).mean().reset_index()
            
            elif self._flag['quantity'] == 'sinkage':
                self._df['time'] = pd.to_datetime(self._df['time'])
                self._df['time'] = self._df['time'] - self._df.loc[0, 'time']
                for x in self._df.index:
                    self._df.loc[x, 'time'] = self._df.loc[x, 'time'].seconds + self._df.loc[x, 'time'].microseconds/1000000
                _ = list(range(self._df.index.size - 5, self._df.index.size))
                self._df.drop(_, inplace=True)
                self._df.rename(columns={'time': 'Time', '.wheel_sinkage': 'Sinkage'}, inplace=True)
        
        elif self._flag['type'] == 'simulation':
            if self._flag['quantity'] == 'force':
                self._df['Fx/Fz'] = self._df["wheel.fx"]/self._df["wheel.fz"]
                self._df['Fx/Fz'].fillna(0, inplace=True)
                self._df['Time'] = self._df['Time'] - self._df.loc[0, 'Time']
                for x in self._df.index:
                    if not -5 <= self._df.loc[x, 'Fx/Fz'] <=5:
                        self._df.drop(x, inplace=True)
                self._df.rename(columns={'wheel.fx': 'Fx', 'wheel.fz': 'Fz'}, inplace=True)
            
            elif self._flag['quantity'] == 'sinkage':
                self._df['Time'] = self._df['Time'] - self._df.loc[0, 'Time']
                
        return self._df
    
    @property
    def plot_data(self):
        self._df = self.clean_data
        if self._flag['quantity'] == 'force':
                plt.plot('Time', 'Fx/Fz', data=self._df, linestyle='-')
        elif self._flag['quantity'] == 'sinkage':
            plt.plot('Time', 'Sinkage', data=self._df, linestyle='-')
        
        plt.show()
    
    def plot_data_compare(self, data=None):
        _ = 10
        fig, ax = plt.subplots()
        
        for i in range(0,5):
            if self._flag['type'] == 'experiment':
                if self._flag['quantity'] == 'force':
                    ax.set(xlabel='Time', ylabel='Fx/Fz', title='Fx/Fz Experiment', autoscale_on=True, xlim=(0,20))
                    ax.plot('Time', 'Fx/Fz', data=data[i], linestyle='-', label=f'SR{_}')
                elif self._flag['quantity'] == 'sinkage':
                    ax.set(xlabel='Time', ylabel='Sinkage', title='Sinkage Experiment', autoscale_on=True)
                    ax.plot('Time', 'Sinkage', data=data[i], linestyle='-', label=f'SR{_}')
            
            elif self._flag['type'] == 'simulation':
                if self._flag['quantity'] == 'force':
                    ax.set(xlabel='Time', ylabel='Fx/Fz', title='Fx/Fz Simulation', autoscale_on=True, xlim=(0,2))
                    ax.plot('Time', 'Fx/Fz', data=data[i], linestyle='-', label=f'SR{_}')
                elif self._flag['quantity'] == 'sinkage':
                    ax.set(xlabel='Time', ylabel='Sinkage', title='Sinkage Simulation', autoscale_on=True)
                    ax.plot('Time', 'Sinkage', data=data[i], linestyle='-', label=f'SR{_}')
            _ = _ + 20
        plt.legend()
        plt.show()
        #return cls
        




if __name__ == '__main__':
    
    data_values = {'type': f'{sys.argv[1]}', 'quantity': f'{sys.argv[2]}'}
    date = 20220615
    SR = [10, 30, 50, 70, 90]
    data = []
    dataFrame = []
    for i in range(0,5):
        data.insert(i, data_frame(date, SR[i], data_values))
        dataFrame.insert(i,data[i].clean_data)
    
    if sys.argv[3] == 'single':
        data[0].plot_data
    
    elif sys.argv[3] == 'compare':
        data[0].plot_data_compare(dataFrame)
    
    