DA6401 Assignment 1 — MLP from Scratch

This project is a Multi-Layer Perceptron (MLP) built with NumPy and trained on the MNIST dataset.  It is then tested on Fashion-MNIST datasets. It uses Weights & Biases (WandB) to track experiments.

---

Project Structure

├── neural_network.py   # MLP, Dense layer, optimizers, loss functions

├── train.py            # Training script with WandB logging

├── inference.py        # Test evaluation, classification report & confusion matrix

├── Data_explore.py     # Dataset exploration and WandB image table logging

├── best_model.npy      # Saved best model weights (generated after training)

├── best_config.json    # Saved best model config (generated after training)



---

 Model Architecture

* Input: 784 neurons (28×28 flattened images)
* Hidden Layers: Configurable number and size
* Output: 10 neurons (Softmax)
* Activations: sigmoid, tanh, relu
* Weight Init: Xavier, random, zero

---

Installation

pip install numpy tensorflow scikit-learn wandb matplotlib seaborn pandas

---

Training:

The model can be trained on the MNIST dataset using the following example code:

python train.py  -d  mnist -e 20  -b 64  -l cce -o momentum -lr 0.001 -nhl 3 -sz 128 -a relu -wi xavier /

 --save_model best_model.npy  --save_config best_config.json -wp DA6401-Assignment1

---

Training Arguments:
| Argument | Flag | Choices | Description |
|---|---|---|---|
| Dataset | `-d` | `mnist`, `fashion_mnist` | Dataset to use |
| Epochs | `-e` | int | Number of training epochs |
| Batch Size | `-b` | int | Mini-batch size |
| Loss | `-l` | `mse`, `cce` | Loss function |
| Optimizer | `-o` | `sgd`, `momentum`, `nag`, `rmsprop` | Optimizer type |
| Learning Rate | `-lr` | float | Learning rate |
| Num Layers | `-nhl` | int | Number of hidden layers |
| Hidden Size | `-sz` | int(s) | Hidden layer sizes (one or many) |
| Activation | `-a` | `sigmoid`, `tanh`, `relu` | Activation function |
| Weight Init | `-wi` | `random`, `xavier`, `zero` | Weight initialization strategy |
| WandB Project | `-wp` | str | WandB project name (default: `DA6401-Assignment1`) |



---

Inference

The evaluation can be done on the test MNIST dataset using the following example code:

python inference.py -d fashion_mnist -e 20 -b 64 -l cce -o momentum -lr 0.001 -nhl 3 -sz 128 -a relu -wi xavier -wp DA6401-Assignment1


Wandb_report
https://wandb.ai/bt25d030-indian-institute-of-technology-madras/DA6401_Assignment1_sweep/reports/DA6401_Assignment_1--VmlldzoxNjExNDU4Mg

Github limk
https://github.com/MaheswarRamS/da6401_assignment_1.git

