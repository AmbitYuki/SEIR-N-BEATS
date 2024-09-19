import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from math import *
import datetime
import matplotlib.dates as mdates
from Dynamic_SEIR_model import *
from helper_fun_epi_model import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

China_population = 1400000000
Hubei_population = 58500000
Zhejiang_population=64567588

Guangdong_population=126012510
Henan_population=99365519
Shandong_population=101527453
Shanghai_population = 24870895
Jiangsu_population = 84748016
Zhejiang_population = 64567588
Anhui_population = 61027171
Shanxi_population=39528999

## Load data
df = pd.read_csv("D:/covid_my/DXYArea.csv")
"""
Data Cleaning 
"""
#df['date']
df['date'] = pd.to_datetime(df['date'])
df = df[df['countryCode']=='China']
# df = df[df['date'] > datetime.datetime(2021, 12, 31)] # first day is 2022-01-01
# df['Days']=(df['date']-datetime.datetime(2022, 1, 1)).map(lambda x: x.days)
#df = df[df['date'] != df['date'].max()] # remove todays' records (since it can be incompleted)

## Dataset preperation
df['R'] = df['province_curedCount'] + df['province_deadCount']
SIR_data = df[['date','countryCode','province','province_confirmedCount', 'province_suspectedCount', 'R',
              ]].rename(columns={"province_suspectedCount": "E"})
# SIR_data = SIR_data.dropna(axis=0,subset=[5])
SIR_data.replace(np.nan, 0, inplace=True)
SIR_data['E'] = SIR_data['E'].astype(int)
SIR_data['Days'] = (SIR_data['date']-datetime.datetime(2020, 1, 22)).map(lambda x: x.days)
# print(SIR_data)
##选择所需省份数据
def get_province_data(df, provinceName: str) -> pandas.core.frame.DataFrame:
    """
    Return time series data of given province
    """
    df = (df[df['province']==provinceName]).drop_duplicates(subset='date', keep='first', inplace=False)
    # df = ((df.set_index('date')).asfreq('D', method='ffill')).reset_index(drop = False)
    df['I'] = df['province_confirmedCount']- df['R']
    return df

# for i in ('Beijing','Guangdong','Shandong','Shaanxi','Henan','Tianjin','Chongqing','Anhui','Fujian','Guangxi','Guizhou','Gansu','Hainan','Hebei','Heilongjiang','Hunan','Jiangsu','Jiangxi','Liaoning','Neimenggu','Ningxia',
# 'Qinghai','Shanxi','Sichuan','Xizang','Xinjiang','Yunnan','Zhejiang'):
# for i in ('Beijing','Hubei'):
i = get_province_data(SIR_data,'Guangdong')
i = i[i['date'] < datetime.datetime(2020,7,1)]
    # plt.plot(i['date'],i['I'])
print(i)
# i.to_csv("D:/下载/n-beats-master新/n-beats-master/examples/data/Guangdong.csv")
