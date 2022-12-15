import pandas as pd
import numpy as np
import datetime as dt              # working with dates

dataset = pd.read_csv('btcusdt_1moth.csv')
dataset['datetime'] = dataset['datetime'].map(lambda x: dt.datetime.fromtimestamp(x / 1000.0))
print(dataset)
