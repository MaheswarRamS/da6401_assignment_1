import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
import argparse
import wandb


def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def loss_and_grad(logits, y_oh, loss_name):
    prob = softmax(logits)
    if loss_name == 'cce':
        prob = np.clip(prob, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.sum(y_oh * np.log(prob), axis=1))
        dloss = prob - y_oh
    else:  # MSE
        loss = np.mean((prob - y_oh) ** 2)
        N = y_oh.shape[1]
        diff = prob - y_oh
        dloss = (2 / N) * prob * (diff - np.sum(diff * prob, axis=1, keepdims=True))
    return loss, dloss


class dense:
    def __init__(self, in_size, out_size, activation='relu', w_init='xavier'):
        self.in_size = in_size
        self.out_size = out_size

        if w_init == 'xavier':
            lim = np.sqrt(2 / (in_size + out_size))
            self.w = np.random.uniform(-lim, lim, (in_size, out_size))
        elif w_init == 'zero':
            self.w = np.zeros((in_size, out_size))
        else:
            self.w = np.random.randn(in_size, out_size) * 0.01

        self.b = np.zeros((1, out_size))

        self.w_grad = None
        self.b_grad = None
        self.x = None
        self.z = None

        # Select activation based upon user input
        if activation == None:
            self.act = lambda x: x
            self.act_grad = lambda z: 1
        elif activation == 'sigmoid':
            self.act = lambda x: 1 / (1 + np.exp(-x))
            self.act_grad = lambda z: self.act(z) * (1 - self.act(z))
        elif activation == 'relu':
            self.act = lambda x: np.maximum(0, x)
            self.act_grad = lambda z: (z > 0).astype(float)
        else:  # tanh
            self.act = lambda x: np.tanh(x)
            self.act_grad = lambda z: 1 - np.tanh(z) ** 2

    # Define forward pass to calculate Z and activation
    def forward(self, x):
        self.x = x
        # linear transformation z = w.x + b
        self.z = np.dot(x, self.w) + self.b
        self.a = self.act(self.z)
        return self.a

    # Define backward method to calculate gradients and store it
    def backward(self, dl_out):
        # Calculate the derivatives for the activation function
        dl_dz = dl_out * self.act_grad(self.z)
        self.w_grad = np.dot(self.x.T, dl_dz) / self.x.shape[0]
        self.b_grad = np.sum(dl_dz, axis=0, keepdims=True) / self.x.shape[0]
        return np.dot(dl_dz, self.w.T)


class NeuralNetwork:
    def __init__(self, in_size, hid_size, out_size, activation='relu', w_init='xavier'):
        self.layers = []
        if isinstance(hid_size, int):
            hid_size = [hid_size]
        size = [in_size] + list(hid_size) + [out_size]
        for i in range(len(size) - 1):
            act = activation if i < len(size) - 2 else None
            self.layers.append(dense(size[i], size[i + 1], act, w_init))
        self.opt_state = [{} for _ in self.layers]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x  # logits

    def backward(self, dl_out):
        for layer in reversed(self.layers):
            dl_out = layer.backward(dl_out)
        return self.get_grad()

    def get_grad(self):
        grad = []
        for layer in self.layers:
            grad += [layer.w_grad, layer.b_grad]
        return grad

    def update_weights(self, lr, opt):
        for i, layer in enumerate(self.layers):
            gw = layer.w_grad
            gb = layer.b_grad
            s = self.opt_state[i]

            if opt == 'sgd':
                layer.w -= lr * gw
                layer.b -= lr * gb

            elif opt == 'momentum':
                beta = 0.9
                s['vw'] = beta * s.get('vw', 0) + gw
                s['vb'] = beta * s.get('vb', 0) + gb
                layer.w -= lr * s['vw']
                layer.b -= lr * s['vb']

            elif opt == 'nag':
                beta = 0.9
                s['vw'] = beta * s.get('vw', 0) + gw
                s['vb'] = beta * s.get('vb', 0) + gb
                layer.w -= lr * (beta * s['vw'] + gw)
                layer.b -= lr * (beta * s['vb'] + gb)

            elif opt == 'rmsprop':
                beta, eps = 0.9, 1e-8
                s['sw'] = beta * s.get('sw', 0) + (1 - beta) * gw ** 2
                s['sb'] = beta * s.get('sb', 0) + (1 - beta) * gb ** 2
                layer.w -= lr * gw / (np.sqrt(s['sw']) + eps)
                layer.b -= lr * gb / (np.sqrt(s['sb']) + eps)

    def train(self, X_train, y_train, loss_name='cce', lr=0.001, opt='sgd', epochs=1, batch_size=32):
        n = X_train.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_train, y_train = X_train[idx], y_train[idx]
            for start in range(0, n, batch_size):
                xb = X_train[start:start + batch_size]
                yb = y_train[start:start + batch_size]
                logits = self.forward(xb)
                _, dl = loss_and_grad(logits, yb, loss_name)
                self.backward(dl)
                self.update_weights(lr, opt)

    def evaluate(self, X, y, loss_name='cce'):
        logits = self.forward(X)
        loss, _ = loss_and_grad(logits, y, loss_name)
        preds = np.argmax(logits, axis=1)
        trues = np.argmax(y, axis=1)
        acc = float(np.mean(preds == trues))
        return loss, acc

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.w.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            if f"W{i}" in weight_dict:
                layer.w = weight_dict[f"W{i}"].copy()
            if f"b{i}" in weight_dict:
                layer.b = weight_dict[f"b{i}"].copy()

    def save(self, path='best_model.npy'):
        params = [(layer.w, layer.b) for layer in self.layers]
        np.save(path, np.array(params, dtype=object), allow_pickle=True)
        return params

    def load(self, path):
        params = np.load(path, allow_pickle=True)
        for i, layer in enumerate(self.layers):
            layer.w = params[i][0]
            layer.b = params[i][1]

# alias for Neural_network
MLP = NeuralNetwork


class optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.v_w = {}
        self.v_b = {}

    def sgd(self, model):
        for layer in model.layers:
            if layer.w_grad is not None:
                layer.w -= self.lr * layer.w_grad
                layer.b -= self.lr * layer.b_grad

    def momentum(self, model, mo=0.9):
        for layer in model.layers:
            if layer.w_grad is not None:
                if layer not in self.v_w:
                    self.v_w[layer] = np.zeros_like(layer.w)
                    self.v_b[layer] = np.zeros_like(layer.b)
                v_w = mo * self.v_w[layer] + self.lr * layer.w_grad
                v_b = mo * self.v_b[layer] + self.lr * layer.b_grad
                layer.w -= v_w
                layer.b -= v_b
                self.v_w[layer] = v_w
                self.v_b[layer] = v_b

    def nag(self, model, mo=0.9):
        for layer in model.layers:
            if layer.w_grad is not None:
                if layer not in self.v_w:
                    self.v_w[layer] = np.zeros_like(layer.w)
                    self.v_b[layer] = np.zeros_like(layer.b)
                v_W_old = self.v_w[layer].copy()
                v_b_old = self.v_b[layer].copy()
                self.v_w[layer] = mo * v_W_old + self.lr * layer.w_grad
                self.v_b[layer] = mo * v_b_old + self.lr * layer.b_grad
                layer.w -= self.v_w[layer] + mo * (self.v_w[layer] - v_W_old)
                layer.b -= self.v_b[layer] + mo * (self.v_b[layer] - v_b_old)

    def rmsprop(self, model, rho=0.9, eps=1e-8):
        for layer in model.layers:
            if layer.w_grad is not None:
                if layer not in self.v_w:
                    self.v_w[layer] = np.zeros_like(layer.w)
                    self.v_b[layer] = np.zeros_like(layer.b)
                self.v_w[layer] = rho * self.v_w[layer] + (1 - rho) * layer.w_grad ** 2
                self.v_b[layer] = rho * self.v_b[layer] + (1 - rho) * layer.b_grad ** 2
                layer.w -= self.lr * layer.w_grad / (np.sqrt(self.v_w[layer]) + eps)
                layer.b -= self.lr * layer.b_grad / (np.sqrt(self.v_b[layer]) + eps)

    def choose_optimizer(self, name, lr):
        self.lr = lr
        step = getattr(self, name.lower(), None)
        if step is None:
            raise ValueError(f"Invalid optimizer name: {name}")
        return step
