import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas as pd
import pandas
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

# China_population = 1400000000
# Hubei_population = 58500000
# Guangdong_population=126012510
# Shanghai_population = 24870895
Beijing_population=21843000
## Load data
df = pd.read_csv("E:/下载/n-beats-master新/n-beats-master/examples/data/shanghai2.csv")

"""
Data Cleaning 
"""
df['date'] = pd.to_datetime(df['date'])
# df = df[df['date'] > datetime.datetime(2019, 12, 7)]  # first day is 2019-12-08
df = df[df['date'] != df['date'].max()] # remove todays' records (since it can be incompleted)

## Dataset preperation
# df['R'] = df['cured'] + df['dead']
SIR_data = df[['date', 'Days', 'I','E', 'R',
              ]]
# SIR_data.tail(3)

# China total

# Use data before 2020-02-14 for train model
Shanghai_df = SIR_data[SIR_data['date'] < datetime.datetime(2022,5, 17)]
# China_total = get_China_total(SIR_data)
# print(China_total)
# China_total.to_csv('date')
Dynamic_SEIR1 = Train_Dynamic_SEIR(epoch =100, data = Shanghai_df ,
                 population = 24870895, rateAl = 1/14, rateIR=1/10, c = 1, b = -5, alpha =0.5)#b = -10, alpha = 0.8

estimation_df = Dynamic_SEIR1.train()#train
print(estimation_df)
# 2.3
test = SIR_data[SIR_data['date'] < datetime.datetime(2022, 5, 17)]
y = test["I"].reset_index(drop = True)
y_pred = estimation_df[:len(test)]['Estimated_Infected'].reset_index(drop = True)
est_beta= estimation_df[:len(test)]['est_beta'].reset_index(drop = True)
err = y-y_pred
# err = pd.DataFrame(err,columns=list('err'))


err = err.reset_index()
# Shanghai_df = Shanghai_df.reset_index()
err.columns = ['index','err']
# print(err)
err_SEIR = pd.concat([estimation_df, err['err'],est_beta],axis=1)
err_SEIR.to_csv("E:/下载/n-beats-master新/n-beats-master/examples/data/shanghai1_SEIR.csv")
# est_beta
print(err_SEIR)
# err_SEIR.to_csv("D:/下载/n-beats-master新/n-beats-master/examples/data/err_SEIR_shanghai1.csv")
est_beta = Dynamic_SEIR1.rateSI
est_alpha = Dynamic_SEIR1.alpha
est_b = Dynamic_SEIR1.b
est_c = Dynamic_SEIR1.c
population = Dynamic_SEIR1.numIndividuals
#
estimation_df.tail(2)
# Dynamic_SEIR1.plot_fitted_beta_R0(China_total)
#
# Dynamic_SEIR1.plot_fitted_result(China_total)

## use the last observation as the initial point in the new SEIR model

# I is the net confirmed cases (total confirmed case - heal - died)
I0 = list(Shanghai_df ['I'])[-1]
R0 = list(Shanghai_df ['R'])[-1]
# suppose the total number of individuals within incubation period is 4 time of current suscepted cases
E0 = list(Shanghai_df ['E'])[-1]
S0 = population - I0 - E0 - R0
#
seir_new = dynamic_SEIR(eons=8, Susceptible=S0, Exposed = E0,
                    Infected=I0, Resistant=R0, rateIR=1/7,
                    rateAl = 1/10, alpha = est_alpha, c = est_c, b = est_b, past_days = Shanghai_df['Days'].max())
result = seir_new.run(death_rate = 0.01) # assume death rate is 2%
seir_new.plot_noSuscep('Dynamic SEIR for China total', 'population', 'Date', starting_point = Shanghai_df['date'].max())
print(result['Infected'])

"""
Calculate MAPE test score using SEIR model result
"""
# test = SIR_data[SIR_data['date'] >= datetime.datetime(2020, 9, 25)]
#
# # y = test["I"].reset_index(drop =
#
# y_pred = result[:len(test)]['Infected'].reset_index(drop = True)
# print(y_pred)
# # y1 = y - y_pred
# test.to_csv('D:/covid_my/SEIR_total1/act.csv')
result.to_csv('E:/下载/n-beats-master新/n-beats-master/examples/data/pre_shanghai1.csv')
# plot_test_data_with_MAPE(test, result,'Infected cases prediction for China total')
# result.to_csv('D:/covid_my/SEIR_total1/seir_total_pre.csv')
# y1.to_csv('D:/covid_my/SEIR_total1/err.csv')