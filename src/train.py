import numpy as np
import argparse
import wandb
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
from ann import NeuralNetwork as MLP, loss_and_grad MLP, optimizer


def load_data(data_s):
    if data_s == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    x_test = x_test.reshape(-1, 784).astype('float32') / 255
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', choices=['mnist', 'fashion_mnist'], required=True)
    p.add_argument('-e', '--epochs', type=int, required=True)
    p.add_argument('-b', '--batch_size', type=int, required=True)
    p.add_argument('-l', '--loss', choices=['mse', 'cce','cross_etropy'], required=True)
    p.add_argument('-o', '--optimizer', choices=['sgd', 'momentum', 'nag', 'rmsprop'], required=True)
    p.add_argument('-lr', '--learning_rate', type=float, required=True)
    p.add_argument('-nhl', '--num_layers', type=int, required=True)
    p.add_argument('-sz', '--hidden_size', type=int, nargs="+", required=True)
    p.add_argument('-a', '--activation', choices=['sigmoid', 'tanh', 'relu'], required=True)
    p.add_argument('-wi', '--weight_init', choices=['random', 'xavier', 'zero'], required=True)
    p.add_argument('-wp', '--wandb_project', type=str, default='DA6401-Assignment1')
    p.add_argument('--save_model', type=str, default='best_model.npy')
    p.add_argument('--save_config', type=str, default='best_config.json')
    return p.parse_args()


def train(args):
    wandb.init(project=args.wandb_project, config=vars(args))
    x_tr, y_tr, x_val, y_val, x_te, y_te = load_data(args.dataset)

    if len(args.hidden_size) == 1:
        hidden_size = [args.hidden_size[0]] * args.num_layers
    else:
        if len(args.hidden_size) != args.num_layers:
            raise ValueError("Invalid number of hidden layers")
        hidden_size = args.hidden_size

    model = MLP(784, hidden_size, 10, args.activation, args.weight_init)
    opt = optimizer(lr=args.learning_rate)
    step = opt.choose_optimizer(args.optimizer, args.learning_rate)
    best_f1 = -1.0
    epoch = -1
    iteration = 0

    for epoch in range(1, (args.epochs + 1)):
        idx = np.random.permutation(x_tr.shape[0])
        total_loss = 0
        nb = 0

        for i in range(0, x_tr.shape[0], args.batch_size):
            bi = idx[i: i + args.batch_size]
            xb = x_tr[bi]
            yb = y_tr[bi]
            logits = model.forward(xb)
            loss, dl = loss_and_grad(logits, yb, args.loss)
            grad = model.backward(dl)
            step(model)
            total_loss += loss
            nb += 1
        avg_loss = total_loss / max(nb, 1)

        if iteration < 50:
            layer1_wgrad = model.layers[0].w_grad
            neuron_log = {}
            for i in range(5):
                neuron_log[f'neuron_{i}_grad'] = float(np.linalg.norm(layer1_wgrad[:, i]))
            wandb.log({'iteration': iteration, **neuron_log})
        iteration += 1

        val_logits = model.forward(x_val)
        val_pred = np.argmax(val_logits, axis=1)
        val_true = np.argmax(y_val, axis=1)
        val_acc = np.mean(val_pred == val_true)
        val_loss, val_dl = loss_and_grad(val_logits, y_val, args.loss)
        prec, rec, f1, _ = precision_recall_fscore_support(val_true, val_pred, average='weighted', zero_division=0)

        # Dead Neuron check
        _ = model.forward(x_val[:1000])
        total_dead = 0
        dead_log = {}
        for i, layer in enumerate(model.layers[:-1]):
            acts = layer.a
            dead = int(np.sum(np.all(acts == 0, axis=0)))
            total_dead += dead
            dead_log[f'dead_layer{i + 1}'] = dead

        layer1_acts = model.layers[0].a.flatten()

        grad_norms = [np.linalg.norm(layer.w_grad) for layer in model.layers[:2] if layer.w_grad is not None]
        wandb.log({'epoch': epoch, 'Train_loss': avg_loss, 'Val_loss': val_loss, 'Val_accuracy': val_acc,
                   'Val_Precision': prec, 'Val_Recall': rec, 'Val_F1': f1,
                   'Grad_norm_layer1': grad_norms[0] if len(grad_norms) > 0 else 0,
                   'Grad_norm_layer2': grad_norms[1] if len(grad_norms) > 1 else 0,
                   'Total_dead_neurons': total_dead, 'Layer1_activation_hist': wandb.Histogram(layer1_acts),
                   **dead_log})
        print(f'epoch: {epoch}, train loss: {avg_loss:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}, val_f1: {f1:.3f}, Total_dead_neuron: {total_dead}')

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            model.save(args.save_model)
            cfg = vars(args).copy()
            cfg['Layers_dim'] = [784] + list(hidden_size) + [10]
            with open(args.save_config, 'w') as f:
                json.dump(cfg, f, indent=2)

    print(f'Best_f1:{best_f1:.3f}, at epoch:{best_epoch}')
    wandb.finish()  # FIXED: was after return (unreachable)
    return best_f1


def main():
    args = parse_arguments()
    train(args)


if __name__ == '__main__':
    main()
