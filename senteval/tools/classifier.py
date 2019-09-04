# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Pytorch Classifier class in the style of scikit-learn
Classifiers include Logistic Regression and MLP
"""

from __future__ import absolute_import, division, unicode_literals

import numpy as np
import copy
from senteval import utils

import torch
from torch import nn
import torch.nn.functional as F

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import hyperopt_changes, hpb
import time

class PyTorchClassifier(object):
    def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64, seed=1111,
                 cudaEfficient=False):
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.inputdim = inputdim
        self.nclasses = nclasses
        self.l2reg = l2reg
        self.batch_size = batch_size
        self.cudaEfficient = cudaEfficient

    def prepare_split(self, X, y, validation_data=None, validation_split=None):
        # Preparing validation data
        assert validation_split or validation_data
        if validation_data is not None:
            trainX, trainy = X, y
            devX, devy = validation_data
        else:
            permutation = np.random.permutation(len(X))
            trainidx = permutation[int(validation_split * len(X)):]
            devidx = permutation[0:int(validation_split * len(X))]
            trainX, trainy = X[trainidx], y[trainidx]
            devX, devy = X[devidx], y[devidx]

        device = torch.device('cpu') if not self.cudaEfficient else torch.device('cuda')

        trainX = torch.from_numpy(trainX).to(device, dtype=torch.float32)
        trainy = torch.from_numpy(trainy).to(device, dtype=torch.int64)
        devX = torch.from_numpy(devX).to(device, dtype=torch.float32)
        devy = torch.from_numpy(devy).to(device, dtype=torch.int64)

        return trainX, trainy, devX, devy

    def fit(self, X, y, validation_data=None, validation_split=None,
            early_stop=True):
        self.nepoch = 0
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0

        # Preparing validation data
        trainX, trainy, devX, devy = self.prepare_split(X, y, validation_data,
                                                        validation_split)

        # Training
        while not stop_train and self.nepoch <= self.max_epoch:
            self.trainepoch(trainX, trainy, epoch_size=self.epoch_size)
            accuracy = self.score(devX, devy)
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                bestmodel = copy.deepcopy(self.model)
            elif early_stop:
                if early_stop_count >= self.tenacity:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel
        return bestaccuracy

    def trainepoch(self, X, y, epoch_size=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + epoch_size):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().to(X.device)

                Xbatch = X[idx]
                ybatch = y[idx]

                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.data.item())
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += epoch_size

    def score(self, devX, devy):
        self.model.eval()
        correct = 0
        #if isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
        if self.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                ybatch = devy[i:i + self.batch_size]
                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                else:
                    if not isinstance(Xbatch, torch.Tensor):
                        Xbatch = torch.tensor(Xbatch)
                        ybatch = torch.tensor(ybatch)
                output = self.model(Xbatch)
                pred = output.data.max(1)[1]
                correct += pred.long().eq(ybatch.data.long()).sum().item()
            accuracy = 1.0 * correct / len(devX)
        return accuracy

    def predict(self, devX):
        self.model.eval()
        if not isinstance(devX, torch.cuda.FloatTensor) and self.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
        elif not isinstance(devX, torch.FloatTensor):
            devX = torch.FloatTensor(devX)
        yhat = np.array([])
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                output = self.model(Xbatch)
                yhat = np.append(yhat,
                                 output.data.max(1)[1].cpu().numpy())
        yhat = np.vstack(yhat)
        return yhat

    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                vals = F.softmax(self.model(Xbatch).data.cpu().numpy())
                if not probas:
                    probas = vals
                else:
                    probas = np.concatenate(probas, vals, axis=0)
        return probas

class HyperPyTorchClassifier(object):
    def __init__(self, inputdim, outputdim, space_search, iter_bayes, seed=1111, cudaEfficient=False):
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.inputdim = inputdim
        self.outputdim = outputdim
        self.iter_bayes = iter_bayes
        self.space_search = space_search
        self.cudaEfficient = cudaEfficient
        self.best_params = None

    def prepare_split(self, X, y, validation_data=None, validation_split=None):
        # Preparing validation data
        assert validation_split or validation_data
        if validation_data is not None:
            trainX, trainy = X, y
            devX, devy = validation_data
        else:
            permutation = np.random.permutation(len(X))
            trainidx = permutation[int(validation_split * len(X)):]
            devidx = permutation[0:int(validation_split * len(X))]
            trainX, trainy = X[trainidx], y[trainidx]
            devX, devy = X[devidx], y[devidx]

        device = torch.device('cpu') if not self.cudaEfficient else torch.device('cuda')

        trainX = torch.from_numpy(trainX).to(device, dtype=torch.float32)
        trainy = torch.from_numpy(trainy).to(device, dtype=torch.int64)
        devX = torch.from_numpy(devX).to(device, dtype=torch.float32)
        devy = torch.from_numpy(devy).to(device, dtype=torch.int64)

        return trainX, trainy, devX, devy

    def fit(self, X, y, validation_data=None, validation_split=None,
            early_stop=True):


        # Preparing validation data
        trainX, trainy, devX, devy = self.prepare_split(X, y, validation_data,
                                                        validation_split)

        # Training
        def train(params):
            self.nepoch = 0
            bestaccuracy = -1
            stop_train = False
            early_stop_count = 0
            self.set_params(params)
            while not stop_train and self.nepoch <= self.max_epoch:
                self.trainepoch(trainX, trainy, epoch_size=self.epoch_size)
                accuracy = self.score(devX, devy)
                if accuracy > bestaccuracy:
                    bestaccuracy = accuracy
                    bestmodel = copy.deepcopy(self.model)
                elif early_stop:
                    if early_stop_count >= self.tenacity:
                        stop_train = True
                    early_stop_count += 1
            self.model = bestmodel
            print ("Loss: {}".format(bestaccuracy))
            return {
                    'loss': -bestaccuracy,
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
        return (-res['loss']*100)

    def trainepoch(self, X, y, epoch_size=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + epoch_size):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().to(X.device)

                Xbatch = X[idx]
                ybatch = y[idx]

                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.data.item())
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += epoch_size

    def score(self, devX, devy):
        self.model.eval()
        correct = 0
        #if isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
        if self.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                ybatch = devy[i:i + self.batch_size]
                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                else:
                    if not isinstance(Xbatch, torch.Tensor):
                        Xbatch = torch.tensor(Xbatch)
                        ybatch = torch.tensor(ybatch)
                output = self.model(Xbatch)
                pred = output.data.max(1)[1]
                correct += pred.long().eq(ybatch.data.long()).sum().item()
            accuracy = 1.0 * correct / len(devX)
        return accuracy

    def predict(self, devX):
        self.model.eval()
        if not isinstance(devX, torch.cuda.FloatTensor) and self.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
        elif not isinstance(devX, torch.FloatTensor):
            devX = torch.FloatTensor(devX)
        yhat = np.array([])
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                output = self.model(Xbatch)
                yhat = np.append(yhat,
                                 output.data.max(1)[1].cpu().numpy())
        yhat = np.vstack(yhat)
        return yhat

    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i:i + self.batch_size]
                vals = F.softmax(self.model(Xbatch).data.cpu().numpy())
                if not probas:
                    probas = vals
                else:
                    probas = np.concatenate(probas, vals, axis=0)
        return probas

    def set_params(self):
        raise 'NotImplemented'


"""
MLP with Pytorch (nhid=0 --> Logistic Regression)
"""

class MLP(PyTorchClassifier):
    def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, cudaEfficient=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cudaEfficient)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """

        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]

        if params["nhid"] == 0:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
            )
            if self.cudaEfficient:
                self.model = self.model.cuda()
        else:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, params["nhid"]),
                nn.Dropout(p=self.dropout),
                nn.Sigmoid(),
                nn.Linear(params["nhid"], self.nclasses),
            )
            if self.cudaEfficient:
                self.model = self.model.cuda()

        self.loss_fn = nn.CrossEntropyLoss()
        if self.cudaEfficient:
            self.loss_fn = self.loss_fn.cuda()
        self.loss_fn.size_average = False

        optim_fn, optim_params = utils.get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg


"""
Bayesian Optimization for MLP with Pytorch/hyperopt (nhid=0 --> Logistic Regression)
"""

class MLPBayesSearch(HyperPyTorchClassifier):
    def __init__(self, inputdim, outputdim, search_params, iter_bayes=100, seed=1111, cudaEfficient=False):
        super(self.__class__, self).__init__(inputdim, outputdim, search_params, iter_bayes, seed, cudaEfficient)

    def set_params(self, params):
        if params['type']['type'] == 'MLP':
            self.nb_layers = 1 if "nb_layers" not in params['type'] else int(params['type']["nb_layers"])
            self.nb_hid = 50 if "nb_hid" not in params['type'] else int(params['type']["nb_hid"])
            if "act_fn" not in params['type'] or params['type']["act_fn"] == 'sigmoid':
                self.act_fn = nn.Sigmoid()
            elif params['type']["act_fn"] == 'tanh':
                self.act_fn = nn.Tanh()
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
            self.model = nn.Sequential(nn.Linear(self.inputdim, self.outputdim))
        else:
            modules = []
            modules.append(nn.Linear(self.inputdim, self.nb_hid))
            for l in np.arange(self.nb_layers-1):
                modules.append(nn.Linear(self.nb_hid, self.nb_hid))
                modules.append(nn.Dropout(p=self.dropout))
                modules.append(self.act_fn)
            modules.append(nn.Linear(self.nb_hid, self.outputdim))

            self.model = nn.Sequential(*modules)

        #print (self.model)
        if self.cudaEfficient:
            self.model = self.model.cuda()

        self.loss_fn = nn.CrossEntropyLoss()
        if self.cudaEfficient:
            self.loss_fn = self.loss_fn.cuda()
        self.loss_fn.size_average = False

        optim_fn, optim_params = utils.get_optimizer(self.optimizer)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg
