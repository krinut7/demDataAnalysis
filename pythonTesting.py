"""Testing python things."""

import pandas as pd
import numpy as np


def func():
    pass


df1 = pd.DataFrame(
    dict(x=np.random.randint(0, 5, 100), y=np.random.randint(0, 5, 100))
)
df2 = pd.DataFrame(
    dict(a=np.random.randint(0, 5, 100), b=np.random.randint(0, 5, 100))
)
df = pd.concat([df1, df2])
foo = df.groupby(df.index).mean()
print(df.to_string())
