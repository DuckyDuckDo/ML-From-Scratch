import numpy as np
import matplotlib.pyplot as plt
import sklearn

class LogisticRegression:
    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward_pass(self, x):
        z = np.dot(x, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return y_pred
    
    def get_loss(self, y_true, y_pred):
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def get_accuracy(self, y_true, y_pred):
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        accuracy = np.mean(y_true == y_pred_class)
        return accuracy

    def train(self, x, y_true):
        num_samples, num_features = x.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for i in range(self.num_iterations):
            y_pred = self.forward_pass(x)
            loss = self.get_loss(y_true, y_pred)
            self.losses.append(loss)

            dw = 1/num_samples * np.dot(x.T, (y_pred - y_true))
            db = 1/num_samples * np.sum(y_pred - y_true)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 10000 == 0:
                print(f"Loss after iteration {i}: {loss}    Accuracy: {self.get_accuracy(y_true, y_pred)}")
        self.plot_loss()

    def predict(self, x):
        y_pred = self.forward_pass(x)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_class)

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title("Loss over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

# Example usage
data = sklearn.datasets.load_breast_cancer()
x = data.data
y = data.target
scaler = sklearn.preprocessing.StandardScaler()
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(learning_rate = 0.001, num_iterations = 100000)
model.train(X_train, y_train)
predictions = model.predict(X_test)
accuracy = model.get_accuracy(y_test, predictions)
print(f"Accuracy: {accuracy}")

