import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn.functional as F  
from scipy.ndimage import rotate, shift
import random

train_file = "../extended_mnist_train.pkl"
test_file = "../extended_mnist_test.pkl"

def augment_image(image, rotation_range=10, shift_range=2):
    img = image.reshape(28, 28)
    angle = random.uniform(-rotation_range, rotation_range)
    img = rotate(img, angle, reshape=False, mode='nearest')
    dx = random.uniform(-shift_range, shift_range)
    dy = random.uniform(-shift_range, shift_range)
    img = shift(img, [dx, dy], mode='nearest')
    return img.flatten()

with open(train_file, "rb") as fp:
    train = pickle.load(fp)
with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data, train_labels = [], []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)

test_data = [image.flatten() for image, _ in test]

augmented_data = []
augmented_labels = []

for image, label in zip(train_data, train_labels):
    augmented_data.append(image)                    
    augmented_labels.append(label)
    augmented_data.append(augment_image(image))    
    augmented_labels.append(label)

X_train = np.array(augmented_data) / 255.0
y_train = np.array(augmented_labels)
X_test = np.array(test_data) / 255.0
Y_train = np.eye(10)[y_train]

class Weights:
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

class Activations:
    def __init__(self, X, z1, a1, z2, y_pred):
        self.X = X
        self.z1 = z1
        self.a1 = a1
        self.z2 = z2
        self.y_pred = y_pred

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z=z-np.max(z,axis=1,keepdims=True)  
    ez=np.exp(z)
    sum_e=np.sum(ez,axis=1,keepdims=True)
    return ez/sum_e

def forward(X, weights):
    z1=np.matmul(X, weights.W1)+weights.b1
    a1=relu(z1)
    z2=np.matmul(a1, weights.W2)+weights.b2
    y_pred=softmax(z2)
    return Activations(X, z1, a1, z2, y_pred)
    
def backward(Y_true, weights, activations, lr):
    batch_size=activations.X.shape[0]
    delta2=(activations.y_pred-Y_true)
    dW2=np.matmul(activations.a1.T, delta2)/batch_size
    db2=np.sum(delta2, axis=0, keepdims=True)/batch_size
    delta1=np.matmul(delta2, weights.W2.T)*(activations.z1 > 0)
    dW1=np.matmul(activations.X.T, delta1)/batch_size
    db1=np.sum(delta1, axis=0, keepdims=True)/batch_size
    weights.W2-=lr*dW2
    weights.b2-=lr*db2
    weights.W1-=lr*dW1
    weights.b1-=lr*db1

    return weights

def compute_loss(y_pred, y_true):
    y_pred_t = torch.tensor(y_pred, dtype=torch.float32)
    y_true_t = torch.tensor(np.argmax(y_true, axis=1), dtype=torch.long) 
    loss = F.cross_entropy(y_pred_t.log(), y_true_t).item()
    return loss 

def compute_accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

def train(X, Y, lr, epochs, batch_size, hidden_dim=100):
    print(f"Training with lr={lr}, batch_size={batch_size}, epochs={epochs}")
    weights=Weights(784, hidden_dim, 10)
    sample_count=X.shape[0]
    best_loss=float('inf')
    patience=3
    wait=0
    for epoch in range(epochs):
        indices = np.arange(sample_count)
        np.random.shuffle(indices)
        X_shuffled, Y_shuffled = X[indices], Y[indices]
        for i in range(0, sample_count, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            activations = forward(X_batch, weights)
            backward(Y_batch, weights, activations, lr)
        activations_all = forward(X, weights)
        acc = compute_accuracy(activations_all.y_pred, Y)
        loss = compute_loss(activations_all.y_pred, Y)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}  Accuracy: {acc:.4f}")
        if loss<best_loss:  
            best_loss=loss
            wait=0
        else:
            wait+=1
            if wait >= patience:
                lr*=0.5
                wait=0
                print(f"reducing learning rate to {lr:.4f}")
    return weights

def predict(X, weights):
    activations = forward(X, weights)
    return np.argmax(activations.y_pred, axis=1)

def save_predictions(predictions):
    df = pd.DataFrame({"ID": np.arange(len(predictions)), "target": predictions})
    df.to_csv("submission.csv", index=False)

weights = train(X_train, Y_train, lr=0.5, epochs=40, batch_size=64)
preds = predict(X_test, weights)
save_predictions(preds)
