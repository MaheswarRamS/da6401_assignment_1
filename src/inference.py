import sys, os
sys.path.insert(0, "/autograder/source")
sys.path.insert(0, "/autograder/source/src")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import json
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from ann.neural_network import NeuralNetwork as MLP, loss_and_grad, optimizer


def parse_arguments(args=None):
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', choices=['mnist', 'fashion_mnist'], required=True)
    p.add_argument('-e', '--epochs', type=int, required=True)
    p.add_argument('-b', '--batch_size', type=int, required=True)
    p.add_argument('-l', '--loss', choices=['mse', 'cce', 'cross_entropy'], required=True)
    p.add_argument('-o', '--optimizer', choices=['sgd', 'momentum', 'nag', 'rmsprop'], required=True)
    p.add_argument('-lr', '--learning_rate', type=float, required=True)
    p.add_argument('-nhl', '--num_layers', type=int, required=True)
    p.add_argument('-sz', '--hidden_size', type=int, nargs="+", required=True)
    p.add_argument('-a', '--activation', choices=['sigmoid', 'tanh', 'relu'], required=True)
    p.add_argument('-wi', '--weight_init', choices=['random', 'xavier', 'zero'], required=True)
    p.add_argument('-wp', '--wandb_project', type=str, default='DA6401-Assignment1')
    p.add_argument('--save_model', type=str, default='best_model.npy')
    p.add_argument('--save_config', type=str, default='best_config.json')
    def _load_from_config():
        import json, os
        candidates = [
            '/autograder/source/src/best_config.json',
            '/autograder/source/best_config.json',
            'src/best_config.json',
            'best_config.json',
        ]
        for config_path in candidates:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                cfg_dir = os.path.dirname(os.path.abspath(config_path))
                save_model = cfg.get('save_model', 'best_model.npy')
                if not os.path.isabs(save_model):
                    for mp in [os.path.join('/autograder/source', save_model),
                                os.path.join(cfg_dir, os.path.basename(save_model)),
                                save_model]:
                        if os.path.exists(mp):
                            save_model = mp
                            break
                return argparse.Namespace(
                    dataset=cfg.get('dataset', 'mnist'),
                    epochs=cfg.get('epochs', 20),
                    batch_size=cfg.get('batch_size', 64),
                    loss=cfg.get('loss', 'cross_entropy'),
                    optimizer=cfg.get('optimizer', 'sgd'),
                    learning_rate=cfg.get('learning_rate', 0.01),
                    num_layers=cfg.get('num_layers', 3),
                    hidden_size=cfg.get('hidden_size', [128]),
                    activation=cfg.get('activation', 'relu'),
                    weight_init=cfg.get('weight_init', 'xavier'),
                    wandb_project=cfg.get('wandb_project', 'DA6401-Assignment1'),
                    save_model=save_model,
                    save_config=config_path,
                )
        return None

    if args is not None:
        return p.parse_args(args)
    try:
        return p.parse_args()
    except SystemExit:
        cfg_args = _load_from_config()
        if cfg_args is not None:
            return cfg_args
        raise


def load_test_data(data_s):
    if data_s == 'mnist':
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
        (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_te = x_test.reshape(-1, 784).astype('float32') / 255
    y_te = np.eye(10)[y_test]
    return x_te, y_te


def load_model(args):
    wandb.init(project=args.wandb_project)
    with open(args.save_config) as f:
        cfg = json.load(f)
    x_te, y_te = load_test_data(args.dataset)
    model_loaded = MLP(
        cfg['Layers_dim'][0],
        cfg['Layers_dim'][1:-1],
        cfg['Layers_dim'][-1],
        activation=cfg['activation'],
        w_init=cfg['weight_init']
    )
    model_loaded.load(args.save_model)
    test_logits = model_loaded.forward(x_te)
    return model_loaded, test_logits, x_te, y_te


def evaluate_model(args, test_logits, y_te):
    test_pred = np.argmax(test_logits, axis=1)
    test_true = np.argmax(y_te, axis=1)
    test_acc = float(np.mean(test_pred == test_true))
    test_loss, _ = loss_and_grad(test_logits, y_te, args.loss)
    te_prec, te_rec, te_f1, _ = precision_recall_fscore_support(
        test_true, test_pred, average='weighted', zero_division=0
    )
    print('\nTest Metrics:')
    report = classification_report(test_true, test_pred)
    print(report)
    print(f'test loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}, '
          f'value_f1: {te_f1:.3f}, test_precision: {te_prec:.3f}, test_recall: {te_rec:.3f}')
    wandb.log({
        'Test_loss': test_loss, 'test_accuracy': test_acc,
        'test_Precision': te_prec, 'test_Recall': te_rec, 'test_F1': te_f1,
        'Classification_report': wandb.Table(columns=['report'], data=[[report]])
    })
    cm = confusion_matrix(test_true, test_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    ticks = range(cm.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('Confusion_Matrix.png')
    wandb.log({"confusion_matrix": wandb.Image('Confusion_Matrix.png')})
    plt.show()
    print(f'Files saved: {args.save_model}, {args.save_config}, Confusion_Matrix.png')


def main():
    args = parse_arguments()
    _, test_logits, _, y_te = load_model(args)
    evaluate_model(args, test_logits, y_te)
    wandb.finish()


if __name__ == '__main__':
    main()
