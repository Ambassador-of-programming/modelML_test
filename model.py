import pandas as pd
import numpy as np
import datetime as dt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold
from ta.momentum import RSIIndicator,StochasticOscillator
from fastai.tabular.all import add_datepart
import plotly.graph_objects as go

#remove warnings
import warnings
warnings.filterwarnings('ignore')

data_new = pd.read_csv('btcusdt_2020_4_13_interval6h.csv')

data_new['Date'] = data_new['Date'].map(lambda x: dt.datetime.fromtimestamp(x / 1000.0))
# data_new.columns = data_new.columns.str.replace('datetime', 'Date')

data_new.reset_index(inplace=True)

add_datepart(data_new, 'Date', drop=False)
data_new.drop('Elapsed', axis=1, inplace=True) # not required for the model

def featurecalculator(share):
  share['EMA_9'] = share['Close'].ewm(9).mean() # exponential moving average of window 9
  share['SMA_5'] = share['Close'].rolling(5).mean() # moving average of window 5
  share['SMA_10'] = share['Close'].rolling(10).mean() # moving average of window 10
  share['SMA_15'] = share['Close'].rolling(15).mean() # moving average of window 15
  share['SMA_20'] = share['Close'].rolling(20).mean() # moving average of window 20
  share['SMA_25'] = share['Close'].rolling(25).mean() # moving average of window 25
  share['SMA_30'] = share['Close'].rolling(30).mean() # moving average of window 30
  EMA_12 = pd.Series(share['Close'].ewm(span=12, min_periods=12).mean())
  EMA_26 = pd.Series(share['Close'].ewm(span=26, min_periods=26).mean())
  share['MACD'] = pd.Series(EMA_12 - EMA_26)    # calculates Moving Average Convergence Divergence
  share['RSI'] = RSIIndicator(share['Close']).rsi() # calculates Relative Strength Index 
  share['Stochastic']=StochasticOscillator(share['High'],share['Low'],share['Close']).stoch() # Calculates Stochastic Oscillator
  pass

featurecalculator(data_new)

def labelencode(share):
  LE=LabelEncoder()
  share['Is_month_end']=LE.fit_transform(share['Is_month_end'])
  share['Is_month_start']=LE.fit_transform(share['Is_month_start'])
  share['Is_quarter_end']=LE.fit_transform(share['Is_quarter_end'])
  share['Is_quarter_start']=LE.fit_transform(share['Is_quarter_start'])
  share['Is_year_end']=LE.fit_transform(share['Is_year_end'])
  share['Is_year_start']=LE.fit_transform(share['Is_year_start'])
  pass

labelencode(data_new)

data_new=data_new.iloc[33:]

data_new.reset_index(drop=True,inplace=True)

data_new = data_new.drop(['Year','High','Low','Open','Volume','Date'],axis=1)

data_new[['EMA_9', 'SMA_5', 'SMA_10', 'SMA_15',
       'SMA_20', 'SMA_25', 'SMA_30', 'MACD', 'RSI', 'Stochastic']] = data_new[['EMA_9', 'SMA_5', 'SMA_10', 'SMA_15',
       'SMA_20', 'SMA_25', 'SMA_30', 'MACD', 'RSI', 'Stochastic']].shift(-1)


test_size = 0.15
valid_size = 0.15
test_split_idx = int(data_new.shape[0] * (1-test_size))
valid_split_idx = int(data_new.shape[0] * (1-(valid_size+test_size)))  

#train test split tcs
train = data_new.loc[:valid_split_idx]
valid = data_new.loc[valid_split_idx+1:test_split_idx]
test = data_new.loc[test_split_idx+1:]


y_train = train['Close']
X_train = train.drop(['Close'], 1)

y_valid = valid['Close']
X_valid = valid.drop(['Close'], 1)

y_test = test['Close']
X_test = test.drop(['Close'], 1)

parameters = {
    'n_estimators': np.arange(100, 1000, 100),
    'learning_rate': np.arange(0.01, 0.1, 0.01),
    'max_depth': np.arange(8, 15, 1),
    'gamma': np.arange(0.005, 0.01, 0.001),
    'random_state': [42],
    'min_child_weight': np.arange(1, 4, 1),
    'subsample': np.arange(0.5, 1, 0.1),
    'colsample_bytree': np.arange(0.1, 1, 0.1),
    'colsample_bylevel': np.arange(0.1, 1, 0.1),
}
kfold=KFold(5)
eval_set = [(X_train, y_train), (X_valid, y_valid)]
model = XGBRegressor(objective='reg:squarederror',n_jobs=-1)
clf = GridSearchCV(model, parameters, cv=kfold, scoring='neg_mean_absolute_error', verbose=0)
clf.fit(X_train, y_train)
print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')


model = XGBRegressor(**clf.best_params_, objective='reg:squarederror',n_jobs=-1)
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
y_pred = model.predict(X_test)
print(mean_absolute_error(y_test,y_pred))

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

print(f'Процент ошибки: {smape(y_test, y_pred)}')


# y_test = list(y_test)
# print(len(y_test))
# print(len(y_pred))

# fig.write_image(f"test.png", scale=5)
