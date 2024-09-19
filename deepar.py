#!/usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
DeepAR Model (Pytorch Implementation)
Paper Link: https://arxiv.org/abs/1704.04110
Author: Jing Wang (jingw2@foxmail.com)
'''

import torch 
from torch import nn
import torch.nn.functional as F 
from torch.optim import Adam

import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import util
from datetime import date
import argparse
from progressbar import *
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Gaussian(nn.Module):

    def __init__(self, hidden_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)
    
    def forward(self, h):
        _, hidden_size = h.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(h).squeeze(0)
        return mu_t, sigma_t

class NegativeBinomial(nn.Module):

    def __init__(self, input_size, output_size):
        '''
        Negative Binomial Supports Positive Count Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial, self).__init__()
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)
    
    def forward(self, h):
        _, hidden_size = h.size()
        alpha_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        return mu_t, alpha_t

def gaussian_sample(mu, sigma):
    '''
    Gaussian Sample
    Args:
    ytrue (array like)
    mu (array like)
    sigma (array like): standard deviation

    gaussian maximum likelihood using log 
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))
    '''
    # likelihood = (2 * np.pi * sigma ** 2) ** (-0.5) * \
    #         torch.exp((- (ytrue - mu) ** 2) / (2 * sigma ** 2))
    # return likelihood
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    ypred = gaussian.sample(mu.size())
    return ypred

def negative_binomial_sample(mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    var = mu + mu * mu * alpha
    ypred = mu + torch.randn(mu.size()) * torch.sqrt(var)
    return ypred

class DeepAR(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, lr=1e-3, likelihood="g"):
        super(DeepAR, self).__init__()

        # network
        self.embed_fun =  nn.Linear(1, 32)
        self.attention_value_ori_func = nn.Linear(32, 1)
        self.output_func = nn.Linear(64,1)
        self.output_activate = nn.Tanh()
        self.input_embed = nn.Linear(1, embedding_size)
        self.encoder = nn.LSTM(embedding_size+input_size, hidden_size, \
                num_layers, bias=True, batch_first=True)

        # self.conv = nn.Sequential(
        #     nn.Conv1d(in_channels=33, out_channels=33, kernel_size=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=1, stride=1))#添加卷积模块

        if likelihood == "g":
            self.likelihood_layer = Gaussian(hidden_size, 1)
        elif likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(hidden_size, 1)
        self.likelihood = likelihood
    
    def forward(self, X, y, Xf):
        '''
        Args:
        X (array like): shape (num_time_series, seq_len, input_size)
        y (array like): shape (num_time_series, seq_len)
        Xf (array like): shape (num_time_series, horizon, input_size)
        Return:
        mu (array like): shape (batch_size, seq_len)
        sigma (array like): shape (batch_size, seq_len)
        '''
        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            Xf = torch.from_numpy(Xf).float()
        num_ts, seq_len, _ = X.size()
        _, output_horizon, num_features = Xf.size()
        ynext = None
        ypred = []
        mus = []
        sigmas = []
        h, c = None, None
        for s in range(seq_len + output_horizon):
            if s < seq_len:
                ynext = y[:, s].view(-1, 1)
                yembed = self.input_embed(ynext).view(num_ts, -1)
                x = X[:, s, :].view(num_ts, -1)
            else:
                yembed = self.input_embed(ynext).view(num_ts, -1)
                x = Xf[:, s-seq_len, :].view(num_ts, -1)

                # x1 = x.reshape(4,1)
                # attention_value_ori = torch.exp(self.attention_value_ori_func(x1)
                # attention_value_format = attention_value_ori.reshape(1, 4).unsqueeze(1)
                # ensemble_flag_format = torch.triu(torch.ones([4, 4]), diagonal=1).permute(1, 0).unsqueeze(0)
                # accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze( -1) + 1e-9
                # each_attention_value = attention_value_format * ensemble_flag_format
                # attention_weight_format = each_attention_value / accumulate_attention_value
                # _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
                # _extend_input_data = x.unsqueeze(1)
                # _extend_input_data = _extend_input_data.unsqueeze(-1)
                # _weighted_input_data = _extend_attention_weight_format * _extend_input_data
                # weighted_output = torch.sum(_weighted_input_data, 2)
                # weighted_output = weighted_output.squeeze(-1)
            x = torch.cat([x, yembed], dim=1) # num_ts, num_features + embedding

            inp = x.unsqueeze(1)
          #添加卷积模块
            # x = inp.permute(0, 2, 1)
            # # x = self.conv(x)
            # inp = x.permute(0, 2, 1)
            if h is None and c is None:
                out, (h, c) = self.encoder(inp) # h size (num_layers, num_ts, hidden_size)
            else:
                out, (h, c) = self.encoder(inp, (h, c))
            hs = h[-1, :, :]

            #attention部分
            
            hs1 = hs.unsqueeze(-1)
            hs = self.embed_fun(hs1)
            reshape_feat = hs.reshape(32,32)
            attention_value_ori = torch.exp(self.attention_value_ori_func(reshape_feat))
            attention_value_format = attention_value_ori.reshape(1, 32).unsqueeze(1)
            ensemble_flag_format = torch.triu(torch.ones([32, 32]), diagonal=1).permute(1, 0).unsqueeze(0)
            accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-9
            each_attention_value = attention_value_format * ensemble_flag_format
            attention_weight_format = each_attention_value / accumulate_attention_value
            _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
            torch.save(attention_weight_format, 'temp_attention_weight.pth')
            _extend_input_data = hs.unsqueeze(1)
            _weighted_input_data = _extend_attention_weight_format * _extend_input_data
            weighted_output = torch.sum(_weighted_input_data, 2)
            _mix_F = torch.cat([hs, weighted_output], dim=-1)
            outputs = self.output_activate(self.output_func(_mix_F))
            outputs = outputs.squeeze(-1)

            # hs = F.relu(hs)
            mu, sigma = self.likelihood_layer(outputs)
            mus.append(mu.view(-1, 1))
            sigmas.append(sigma.view(-1, 1))
            if self.likelihood == "g":
                ynext = gaussian_sample(mu, sigma)
            elif self.likelihood == "nb":
                alpha_t = sigma
                mu_t = mu
                ynext = negative_binomial_sample(mu_t, alpha_t)
            # if without true value, use prediction
            if s >= seq_len - 1 and s < output_horizon + seq_len - 1:
                ypred.append(ynext)
        ypred = torch.cat(ypred, dim=1).view(num_ts, -1)
        mu = torch.cat(mus, dim=1).view(num_ts, -1)
        sigma = torch.cat(sigmas, dim=1).view(num_ts, -1)
        return ypred, mu, sigma
    
def batch_generator(X, y, num_obs_to_train, seq_len, batch_size,t):
    '''
    Args:
    X (array like): shape (num_samples, num_features, num_periods)
    y (array like): shape (num_samples, num_periods)
    num_obs_to_train (int):
    seq_len (int): sequence/encoder/decoder length
    batch_size (int)
    '''
    num_ts, num_periods, _ = X.shape
    if num_ts < batch_size:
        batch_size = num_ts
    # t = random.choice(range(num_obs_to_train, num_periods-seq_len))
    batch = random.sample(range(num_ts), batch_size)
    X_train_batch = X[batch, t-num_obs_to_train:t, :]
    y_train_batch = y[batch, t-num_obs_to_train:t]
    Xf = X[batch, t:t+seq_len]
    yf = y[batch, t:t+seq_len]
    return X_train_batch, y_train_batch, Xf, yf

def train(
    X, 
    y,
    args
    ):
    '''
    Args:
    - X (array like): shape (num_samples, num_features, num_periods)
    - y (array like): shape (num_samples, num_periods)
    - epoches (int): number of epoches to run
    - step_per_epoch (int): steps per epoch to run
    - seq_len (int): output horizon
    - likelihood (str): what type of likelihood to use, default is gaussian
    - num_skus_to_show (int): how many skus to show in test phase
    - num_results_to_sample (int): how many samples in test phase as prediction
    '''
    num_ts, num_periods, num_features = X.shape
    model = DeepAR(num_features, args.embedding_size, 
        args.hidden_size, args.n_layers, args.lr, args.likelihood)
    optimizer = Adam(model.parameters(), lr=args.lr)
    random.seed(2)
    # select sku with most top n quantities 
    Xtr, ytr, Xte, yte = util.train_test_split(X, y)
    losses = []
    cnt = 0

    yscaler = None
    if args.standard_scaler:
        yscaler = util.StandardScaler()
    elif args.log_scaler:
        yscaler = util.LogScaler()
    elif args.mean_scaler:
        yscaler = util.MeanScaler()
    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)

    # training
    model.train()
    seq_len = args.seq_len
    num_obs_to_train = args.num_obs_to_train
    progress = ProgressBar()
    num_ts, num_periods, _ = Xtr.shape
    # for epoch in progress(range(args.num_epoches)):
    for epoch in progress(range(args.num_epoches)):
        # print("Epoch {} starts...".format(epoch))
        # for t in range(args.step_per_epoch):
        for t in range(num_obs_to_train, num_periods-seq_len):
            Xtrain, ytrain, Xf, yf = batch_generator(Xtr, ytr, num_obs_to_train, seq_len, args.batch_size,t)
            Xtrain_tensor = torch.from_numpy(Xtrain).float()
            ytrain_tensor = torch.from_numpy(ytrain).float()
            Xf = torch.from_numpy(Xf).float()  
            yf = torch.from_numpy(yf).float()
            ypred, mu, sigma = model(Xtrain_tensor, ytrain_tensor, Xf)
            # ypred_rho = ypred
            # e = ypred_rho - yf
            # loss = torch.max(rho * e, (rho - 1) * e).mean()
            ## gaussian loss
            ytrain_tensor = torch.cat([ytrain_tensor, yf], dim=1)
            if args.likelihood == "g":
                loss = util.gaussian_likelihood_loss(ytrain_tensor, mu, sigma)
            elif args.likelihood == "nb":
                loss = util.negative_binomial_loss(ytrain_tensor, mu, sigma)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
    
    # test
    model.eval()
    mape_list = []
    # select skus with most top K
    # X_test = Xte[:, -seq_len-num_obs_to_train:-seq_len, :].reshape((num_ts, -1, num_features))
    # Xf_test = Xte[:, -seq_len:, :].reshape((num_ts, -1, num_features))
    # y_test = yte[:, -seq_len-num_obs_to_train:-seq_len].reshape((num_ts, -1))
    # yf_test = yte[:, -seq_len:].reshape((num_ts, -1))

    X_test = Xte[:, :num_obs_to_train, :].reshape((num_ts, -1, num_features))
    Xf_test = Xte[:, num_obs_to_train:num_obs_to_train+seq_len, :].reshape((num_ts, -1, num_features))
    y_test = yte[:, :num_obs_to_train].reshape((num_ts, -1))
    yf_test = yte[:, num_obs_to_train:num_obs_to_train+seq_len].reshape((num_ts, -1))
    if yscaler is not None:
        y_test = yscaler.transform(y_test)
    result = []
    n_samples = args.sample_size
    for _ in tqdm(range(n_samples)):
        y_pred, _, _ = model(X_test, y_test, Xf_test)
        y_pred = y_pred.data.numpy()
        if yscaler is not None:
            y_pred = yscaler.inverse_transform(y_pred)
        result.append(y_pred.reshape((-1, 1)))

    result = np.concatenate(result, axis=1)
    # print(result.shape)
    # print(yf_test)
    p50 = np.quantile(result, 0.5, axis=1)
    p90 = np.quantile(result, 0.9, axis=1)
    p10 = np.quantile(result, 0.1, axis=1)
    p60 = np.quantile(result, 0.6, axis=1)
    p80 = np.quantile(result, 0.8, axis=1)
    p70 = np.quantile(result, 0.7, axis=1)
    p25 = np.quantile(result, 0.25, axis=1)
    p40 = np.quantile(result, 0.40, axis=1)
    # print(yf_test)
    print(p50)
    print(p70)
    print(p80)
    print(p40)

    mape = util.MAPE(yf_test, p80)
    print("P50 MAPE: {}".format(mape))

    yf_test = yf_test.squeeze(0)
    print(mean_absolute_error(yf_test, p80))
    print(np.sqrt(mean_squared_error(yf_test, p80)))
    mape_list.append(mape)
    # print(mape_list)

    if args.show_plot:

        plt.figure(1, figsize=(15, 5),dpi=250)
        plt.plot([k + seq_len + num_obs_to_train - seq_len \
            for k in range(seq_len)], p80, "r-")
        plt.fill_between(x=[k + seq_len + num_obs_to_train - seq_len for k in range(seq_len)], \
            y1=p50, y2=p90, alpha=0.5)
        # plt.title('Uncertainty of prediction')
        yplot = yte[-1, -seq_len-num_obs_to_train:]
        plt.plot(range(len(yplot)), yplot, "k-")
        plt.legend(["P80 quantile", "P50-P90 quantile" ,"true"], loc="upper left")
        ymin, ymax = plt.ylim()
        plt.vlines(seq_len + num_obs_to_train - seq_len, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        plt.ylim(ymin, ymax)
        plt.xlabel("time")
        # plt.savefig('电力负荷预测图.svg', bbox_inches='tight', dpi=250, pad_inches=0.0)
        plt.ylabel("value")
        plt.show()
    return losses, mape_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoches", "-e", type=int, default=2)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=2)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", "-nl", type=int, default=2)
    parser.add_argument("--hidden_size", "-hs", type=int, default=32)
    parser.add_argument("--embedding_size", "-es", type=int, default=32)
    parser.add_argument("--likelihood", "-l", type=str, default="g")
    parser.add_argument("--seq_len", "-sl", type=int, default=7)
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=7)
    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=1)
    parser.add_argument("--show_plot", "-sp", action="store_true",default=True)
    parser.add_argument("--run_test", "-rt", action="store_true",default=True)
    parser.add_argument("--standard_scaler", "-ss", action="store_true",default=True)
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--sample_size", type=int, default=100)

    args = parser.parse_args()
    print(args)
    if args.run_test:

        data_path = util.get_data_path()
        # data = pd.read_csv(os.path.join(data_path, "psi日期合并.csv"), parse_dates=["date"])
        data = pd.read_csv(os.path.join(data_path, "LD_MT200_hour_fill.csv"), parse_dates=["date"])
        # data = pd.read_csv(os.path.join(data_path, "COMED_hourly.csv"), parse_dates=["date"])
        # data["year"] = data["datetime"].apply(lambda x: x.year)
        # data["day_of_week"] = data["datetime"].apply(lambda x: x.dayofweek)
        # print(data["day_of_week"])
        # print(data["year"])
        datestr = '2014, 1, 1'
        datestr1 = '2018, 8, 3'
        # data = data.loc[(data["date"] >= '2016-2-7-18:00') & (data["date"] <= '2019-11-6-14:00')]
        # data = data.loc[(data["date"] >= date(2014, 1, 1)) & (data["date"] <= date(2018, 8, 3))]
        data = data.loc[(data["date"] >= datestr) & (data["date"] <= datestr1)]
        # data = data.loc[(data["date"] >= '2011-1-1 1:00') & (data["date"] <= '2018-8-3 0:00')]
        # features = ["hour", "day_of_week"]
        # # features = ["hour", "day_of_week", "DAYTON"]
        # hours = pd.get_dummies(data["hour"])
        # dows = pd.get_dummies(data["day_of_week"])
        # fea1 = data["south"]
        # # dows = data["day_of_week"]
        # fea2 = data["north"]
        # fea3 = data["east"]
        # fea4 = data["central"]
        # fea5 = data["west"]
        # fea1 = data["humidity"]
        # # dows = data["day_of_week"]
        # fea2 = data["wind_speed"]
        # fea3 = data["meanpressure"]

        fea1 = data["East"]
        # dows = data["day_of_week"]
        fea2 = data["DAYTON"]
        fea3 = data["AEP"]
        fea4 = data["DUQ"]
        fea5 = data["DOM"]


        # fea1 = data["hour"]
        # fea2 = data["day_of_week"]
        # fea3 = data["year"]
        # X = np.c_[np.asarray(fea1)]
        # X = np.c_[np.asarray(fea1)]
        # X = np.c_[np.asarray(fea1), np.asarray(fea2), np.asarray(fea3),np.asarray(fea4),np.asarray(fea5)]
        X = np.c_[np.asarray(fea1), np.asarray(fea2), np.asarray(fea3), np.asarray(fea4), np.asarray(fea5)]
        num_features = X.shape[1]
        num_periods = len(data)

        X = np.asarray(X).reshape((-1, num_periods, num_features))
        y = np.asarray(data["MT_200"]).reshape((-1, num_periods))
        print(X.shape)

        # print(y)
        # X = np.tile(X, (10, 1, 1))
        # y = np.tile(y, (10, 1))
        losses, mape_list = train(X, y, args)
        if args.show_plot:
            plt.plot(range(len(losses)), losses, "k-")
            plt.xlabel("Period")
            plt.ylabel("Loss")
            plt.show()
