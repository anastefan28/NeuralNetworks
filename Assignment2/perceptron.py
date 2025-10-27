import numpy as np
import pickle
import pandas as pd

train_file = "../extended_mnist_train.pkl"
test_file = "../extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)
with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)

test_data = []
for image, label in test:
    test_data.append(image.flatten())

X_train=np.array(train_data)/255
y_train=np.array(train_labels)
X_test=np.array(test_data)/255 
Y_train=np.eye(10)[y_train]

def initialize_parameters(input_dim=784, output_dim=10):
    std=1/np.sqrt(input_dim)
    W=np.random.randn(input_dim, output_dim)*0.05
    b=np.zeros((1,output_dim))
    return W,b

def softmax(z):
    z=z-np.max(z,axis=1,keepdims=True)  
    ez=np.exp(z)
    sum_e=np.sum(ez,axis=1,keepdims=True)
    return ez/sum_e

def forward(X,W,b):
    z=np.matmul(X,W)+b
    return softmax(z)

def backward(X, W, b, lr, y_pred, y_true):
    batch_size=X.shape[0]
    W+=lr*np.matmul(np.transpose(X),(y_true-y_pred))/batch_size
    b+=lr*np.sum(y_true-y_pred,axis=0,keepdims=True)/batch_size
    return W,b

def train(X, Y, lr=0.7, epochs=100, batch_size=128):
    print(lr, "learning rate;", batch_size, "batch size;", epochs, "epochs;")
    W,b=initialize_parameters(784,10)
    sample_count=X.shape[0]
    for epoch in range(epochs):
        if (epoch+1)%15==0:
            lr*=0.98
        indices=np.arange(sample_count)
        np.random.shuffle(indices)
        X_shuffled=X[indices]
        Y_shuffled=Y[indices]

        for i in range(0, sample_count, batch_size):
            X_batch=X_shuffled[i:i+batch_size]
            Y_batch=Y_shuffled[i:i+batch_size]
            y_pred=forward(X_batch, W, b)
            W,b=backward(X_batch, W, b, lr, y_pred,Y_batch)
        y_pred_full=forward(X, W, b)
        acc=np.mean(np.argmax(y_pred_full, axis=1)==np.argmax(Y, axis=1))
        print(f"Epoch {epoch+1} Accuracy: {acc:.2f}")
    return W,b


W,b=train(X_train,Y_train)

def predict(X, W, b):
    y_pred=forward(X, W, b)
    return np.argmax(y_pred, axis=1)

predictions=predict(X_test, W, b)
predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)
