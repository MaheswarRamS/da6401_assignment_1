import numpy as np
import argparse
import json
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import argparse
from neural_network import MLP, loss_and_grad


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dataset', choices=['mnist', 'fashion_mnist'], required=True)
    p.add_argument('-e', '--epochs', type=int, required=True)
    p.add_argument('-b', '--batch_size', type=int, required=True)
    p.add_argument('-l', '--loss', choices=['mse', 'cce'], required=True)
    p.add_argument('-o', '--optimizer', choices=['sgd', 'momentum', 'nag', 'rmsprop'], required=True)
    p.add_argument('-lr', '--learning_rate', type=float, required=True)
    p.add_argument('-nhl', '--num_layers', type=int, required=True)
    p.add_argument('-sz', '--hidden_size', type=int, nargs="+", required=True)  
    p.add_argument('-a', '--activation', choices=['sigmoid', 'tanh', 'relu'], required=True)
    p.add_argument('-wi', '--weight_init', choices=['random', 'xavier', 'zero'], required=True)
    p.add_argument('-wp','--wandb_project', type=str, default='DA6401-Assignment1')
    p.add_argument('--save_model', type=str, default='best_model.npy')
    p.add_argument('--save_config', type=str, default='best_config.json')
    return p.parse_args()

def load_test_data(data_s):  
    if data_s == 'mnist':
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_te = x_test.reshape(-1,784).astype('float32')/255
    y_te = np.eye(10)[y_test]
    return x_te, y_te

# Test inference + Report
def inference_metrics(args):
   
    with open (args.save_config) as f:
      cfg = json.load(f)

    x_te,y_te = load_test_data(args.dataset)
    model_loaded = MLP(cfg['Layers_dim'][0],cfg['Layers_dim'][1:-1],cfg['Layers_dim'][-1], activation = cfg['activation'], w_init = cfg['weight_init'])
    model_loaded.load(args.save_model)  
    test_logits = model_loaded.forward(x_te)
    test_pred = np.argmax(test_logits,axis=1)
    test_true = np.argmax(y_te,axis=1)
    test_acc = float(np.mean(test_pred==test_true))
    test_loss,_ = loss_and_grad(test_logits, y_te, args.loss) 
    te_prec,te_rec,te_f1,_ = precision_recall_fscore_support(test_true,test_pred, average='weighted',zero_division=0)
    print('\nTest Metrics:')
    report = classification_report(test_true,test_pred)
    print(report)
    print(f'test loss : {test_loss:3f},test_accuracy:{test_acc:3f},value_f1:{te_f1:3f},test_precesion:{te_prec:3f},test_recall:{te_rec:3f}')
    wandb.log({'Test_loss':test_loss, 'test_accuracy': test_acc,'test_Precision':te_prec, 'test_Recall':te_rec, 'test_F1': te_f1, 'Classification_report': wandb.Table(columns=['report'],data=[[report]]) })
  

# Confusion matrix
    cm = confusion_matrix(test_true,test_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm,annot=True, fmt='d',cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('Confusion_Matrix.png')
    wandb.log({"confusion_matrix": wandb.Image('Confusion_Matrix.png')})
    plt.show()
    print(f'Files saved: {args.save_model},{args.save_config}, Confusion_Matrix.png')
wandb.finish()



def main():
    args = parse_args()
    inference_metrics(args)

if __name__ == '__main__':
  main()