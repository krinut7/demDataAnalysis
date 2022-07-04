"""Testing python things."""

import pandas as pd


class dance:

    def __init__(self):
        pass

    def function1(self):
        d = {'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]}
        df = pd.DataFrame(data=d)
        msg = 'Krishna'
        print(f'df in function1: {id(df)}')
        return df

    def function2(self, df2):
        df = df2.rename(columns={'col1': 'c1', 'col2': 'c2', 'col3': 'c3'})
        print(f'df in function2: {id(df)}')
        msg = 'Krishna sings'
        return df


if __name__ == '__main__':

    hello = dance()

    # print(hello.function1())
    print(hello.function2(hello.function1()))
