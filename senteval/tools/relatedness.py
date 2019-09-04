# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Semantic Relatedness (supervised) with Pytorch
"""
from __future__ import absolute_import, division, unicode_literals

import copy
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from senteval import utils

from scipy.stats import pearsonr

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import hyperopt_changes, hpb
import time


class HyperRelatednessPytorch(object):
    # Can be used for SICK-Relatedness, and STS14
    def __init__(self, train, valid, test, devscores, config):
        # fix seed
        np.random.seed(config['config']['seed'])
        torch.manual_seed(config['config']['seed'])
        #assert torch.cuda.is_available(), 'torch.cuda required for Relatedness'
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['config']['seed'])

        self.train = train
        self.valid = valid
        self.test = test
        self.devscores = devscores

        self.inputdim = train['X'].shape[1]
        self.outputdim = config['config']['outputdim']
        self.seed = config['config']['seed']
        self.early_stop = True

        self.space_search = config['space_search']
        self.iter_bayes = config['iter_bayes']
        self.cudaEfficient = False if 'cudaEfficient' not in config['config'] else \
            config['config']['cudaEfficient']
        self.modelname = 'MLP'# get_classif_name(self.classifier_config, self.usepytorch)
        self.config = config

        self.model = None

    def set_params(self, params):

        if params['type']['type'] == 'MLP':
            self.nb_layers = 1 if "nb_layers" not in params['type'] else int(params['type']["nb_layers"])
            self.nb_hid = 50 if "nb_hid" not in params['type'] else int(params['type']["nb_hid"])
            if "act_fn" not in params['type'] or params['type']["act_fn"] == 'sigmoid':
                self.act_fn = nn.Sigmoid()
            elif params['type']["act_fn"] == 'tanh':
                self.act_fn = nn.Tanh()
            elif params['type']["act_fn"] == 'relu':
                self.act_fn = nn.ReLU()
            elif params['type']["act_fn"] == 'elu':
                self.act_fn = nn.ELU(alpha=1.0)
        self.optimizer = "adam" if "optimizer" not in params else params["optimizer"]
        self.tenacity = 5 if "tenacity" not in params else int(params["tenacity"])
        self.epoch_size = 4 if "epoch_size" not in params else int(params["epoch_size"])
        self.max_epoch = 100 if "max_epoch" not in params else int(params["max_epoch"])
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 128 if "batch_size" not in params else int(params["batch_size"])
        self.l2reg = 0.02 if "l2reg" not in params else params["l2reg"]

        # set model
        if params["type"]['type'] == 'LogisticRegression':
            self.model = nn.Sequential(nn.Linear(self.inputdim, self.outputdim), nn.Softmax(dim=-1))
        elif params["type"]['type'] == 'MLP':
            modules = []
            modules.append(nn.Linear(self.inputdim, self.nb_hid))
            for l in np.arange(self.nb_layers-1):
                modules.append(nn.Linear(self.nb_hid, self.nb_hid))
                modules.append(nn.Dropout(p=self.dropout))
                modules.append(self.act_fn)
            modules.append(nn.Linear(self.nb_hid, self.outputdim))
            modules.append(nn.Softmax(dim=-1))
            self.model = nn.Sequential(*modules)

        self.loss_fn = nn.MSELoss()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        self.loss_fn.size_average = False
        optim_fn, optim_params = utils.get_optimizer(self.optimizer)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg

    def prepare_data(self, trainX, trainy, devX, devy, testX, testy):
        # Transform probs to log-probs for KL-divergence
        trainX = torch.from_numpy(trainX).float()
        trainy = torch.from_numpy(trainy).float()
        devX = torch.from_numpy(devX).float()
        devy = torch.from_numpy(devy).float()
        testX = torch.from_numpy(testX).float()
        testY = torch.from_numpy(testy).float()
        if torch.cuda.is_available():
            for x in [trainX, trainy, devX, devy, testX, testY]:
                x = x.cuda()

        return trainX, trainy, devX, devy, testX, testy

    def run(self):
        r = np.arange(1, 6)

        # Preparing data
        trainX, trainy, devX, devy, testX, testy = self.prepare_data(
            self.train['X'], self.train['y'],
            self.valid['X'], self.valid['y'],
            self.test['X'], self.test['y'])

        # Training
        def train(params):
            r = np.arange(1, 6)
            self.nepoch = 0
            bestpr = -1
            early_stop_count = 0
            stop_train = False
            self.set_params(params)
            while not stop_train and self.nepoch <= self.max_epoch:
                self.trainepoch(trainX, trainy, nepoches=self.epoch_size)
                yhat = np.dot(self.predict_proba(devX), r)
                pr = pearsonr(yhat, self.devscores)[0]
                pr = 0 if pr != pr else pr  # if NaN bc std=0
                # early stop on Pearson
                if pr > bestpr:
                    bestpr = pr
                    bestmodel = copy.deepcopy(self.model)
                elif self.early_stop:
                    if early_stop_count >= 3:
                        stop_train = True
                    early_stop_count += 1
            self.model = bestmodel
            print ("Loss: {}".format(bestpr))
            return {
                    'loss': -bestpr,
                    'status': STATUS_OK,
                    # -- store other results like this
                    'eval_time': time.time(),
                    'params': params,
                    }

        # Bayesian Optimization
        trials = Trials()

        best_params = fmin(train,
            space=self.space_search,
            algo=tpe.suggest,
            max_evals=self.iter_bayes,
            trials=trials)

        self.best_params = space_eval(self.space_search, best_params)
        print (self.best_params)

        # retrain model for best_params (change self.model)
        res = train(self.best_params)

        yhat = np.dot(self.predict_proba(testX), r)

        return -res['loss'], yhat, best_params

    def trainepoch(self, X, y, nepoches=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + nepoches):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.from_numpy(permutation[i:i + self.batch_size]).long()
                if torch.cuda.is_available():
                    idx = idx.cuda()
                Xbatch = X[idx]
                ybatch = y[idx]
                output = self.model(Xbatch.cuda())
                # loss
                loss = self.loss_fn(output, ybatch.cuda())
                all_costs.append(loss.item())
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += nepoches

    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                if len(probas) == 0:
                    probas = self.model(Xbatch).data.cpu().numpy()
                else:
                    probas = np.concatenate((probas, self.model(Xbatch).data.cpu().numpy()), axis=0)
        return probas
