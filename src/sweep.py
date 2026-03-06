import numpy as np
import wandb
from ann import MLP, optimizer, load_data, loss_and_grad

sweep_config = {
                 "method" : "bayes",
                 "metric" : {
                    "name": "val_accuracy",
                    "goal": "maximize"
                  },
                 "parameters":{
                  "dataset": {"value": "mnist" },    
                  "epochs" : {"value": 10},
                  "loss"   : {"values": ["cce","mse"]},
            "learning_rate": {"values": [0.0001,0.001,0.01]},
            "batch_size"   : {"values": [16,32,64]},
             "optimizer"   : {"values": ["sgd","momentum","nag","rmsprop"]},
             "num_layers"  : {"values": [3,4,5,6]},
             "hidden_size" : {"values": [32,64,128]},
             "activation"  : {"values": ["sigmoid","tanh","relu"]},
             "w_init" : {"values": ["xavier","random","zero"]},
                 }
 
}

def sweep_train():
    with wandb.init():
        cfg = wandb.config

        x_train,y_train,x_val,y_val,x_test,y_test = load_data(cfg.dataset)
        hidden_layers = [cfg.hidden_size]* cfg.num_layers
        model = MLP(in_size=784, hid_size = hidden_layers, out_size = 10, activation = cfg.activation, w_init = cfg.w_init) 
        opt = optimizer(lr = cfg.learning_rate)
        step = getattr(opt, cfg.optimizer)
                
        for epo in range(cfg.epochs):
            idx = np.random.permutation(x_train.shape[0])
            total_loss = 0
            num_batches = 0

            for i in range(0,x_train.shape[0],cfg.batch_size):
                batch_idx =  idx[i: i+ cfg.batch_size]
                x_batch = x_train [batch_idx]
                y_batch = y_train[batch_idx]
                logits = model.forward(x_batch)
                loss, dl = loss_and_grad(logits,y_batch, cfg.loss)
                model.backward(dl)
                step(model)
                total_loss += loss
                num_batches += 1
            avg_train_loss = total_loss/ max(num_batches,1)
            val_logits = model.forward(x_val)
            val_pred = np.argmax(val_logits, axis=1)
            val_true = np.argmax(y_val,axis=1)
            val_acc = np.mean(val_pred==val_true)
            val_loss, val_dl = loss_and_grad(val_logits, y_val, cfg.loss)
            test_logits = model.forward(x_test)
            test_pred = np.argmax(test_logits, axis=1)
            test_true = np.argmax(y_test,axis=1)
            test_acc = np.mean(test_pred==test_true)
            test_loss, test_dl = loss_and_grad(test_logits, y_test, cfg.loss)

            wandb.log({
                    "epochs" : epo,
                    "train_loss": avg_train_loss,
                   "val_loss" : val_loss,
                   "test_loss": test_loss,
                   "val_accuracy": val_acc,
                   "test_accuracy": test_acc
                   })
            print(f'epochs : {epo+1}/{cfg.epochs}, train_loss: {avg_train_loss:3f}, val_loss:{val_loss:3f}, test_loss:{test_loss:3f}, val_accuracy : {val_acc:3f}, test_accuracy:{test_acc:3f}')


if __name__ == '__main__':
     sweep_id = wandb.sweep(sweep_config, project= 'DA6401_Assignment1_sweep')
     print(f'Sweep ID Create: {sweep_id}')
     print('Starting 100 Runs ...')

     wandb.agent( sweep_id, function = sweep_train, count = 100)

     print('All runs completed')
