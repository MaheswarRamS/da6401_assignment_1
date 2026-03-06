import numpy as np 
import wandb
import tensorflow as tf

data = tf.keras.datasets.mnist.load_data()
x_train = data[0][0]
y_train = data[0][1]
y_train = np.array(y_train).flatten()

wandb.init(project= 'DA6401_Assignment_1_Data_Explore')

table = wandb.Table(columns=['Class','S1','S2','S3','S4','S5'])

for i in range(10):
    idx = np.where(y_train == i)[0][:5]
    imgs = [wandb.Image(x_train[j]) for j in idx]
    table.add_data(str(i),*imgs)

wandb.log({'Sample_Images':table})
wandb.finish()