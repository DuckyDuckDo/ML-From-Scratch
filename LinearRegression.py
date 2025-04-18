import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate, num_iterations, l2_lambda = None):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.l2_lambda = l2_lambda
        self.weights = None
        self.bias = None
        self.losses = []

    def forward_pass(self, x):
        return np.dot(x, self.weights) + self.bias
    
    def get_loss(self, y_true, y_pred):

        # l2_lambda is for ridge regression, loss function changes if regularization is added
        if not self.l2_lambda:
            return np.mean((y_true - y_pred) ** 2)
        else:
            return np.mean((y_true - y_pred) ** 2) + self.l2_lambda * np.sum(self.weights ** 2) 
    
    def get_r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) **2)
        return 1 - (ss_residual / ss_total)
    
    def train(self, x, y_true):
        num_samples, num_features = x.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Loop through iterations and train, updating weights and bias
        for i in range(self.num_iterations):
            y_pred = self.forward_pass(x)
            loss = self.get_loss(y_true, y_pred)
            self.losses.append(loss)
            r2_score = self.get_r2_score(y_true, y_pred)
            
            # Gradient for weights with and without regularization
            if self.l2_lambda:
                dw = (-2/num_samples) * np.dot(x.T, (y_true - y_pred)) + 2 * self.l2_lambda * self.weights
            else:
                dw = (-2/num_samples) * np.dot(x.T, (y_true - y_pred))
            
            # Gradient for the bias remains the same regardless of regularization
            db = (-2/num_samples) * np.sum(y_true - y_pred)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 25 == 0:
                print(f"Iteration {i}, Loss: {loss}, R2 Score: {r2_score}")
        
        self.plot_loss()
    
    def plot_loss(self):
        plt.figure()
        plt.plot(self.losses)
        plt.title("Loss over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()


# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Get data
    x, y = make_regression(n_samples = 1000, n_features = 1, noise = 10, random_state = 42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train model
    model = LinearRegression(learning_rate = 0.01, num_iterations = 400, l2_lambda = 0.01)
    model.train(x_train, y_train)
    y_pred = model.forward_pass(x_test)
    r2_score = model.get_r2_score(y_test, y_pred)
    print(f"R2 Score on test set: {r2_score}")

    # Plotting predictions vs actual values
    plt.figure()
    plt.scatter(x_test, y_pred, color = "red", label = "Predicted")
    plt.scatter(x_test, y_test, color = "blue", label = "Actual")
    plt.title("Regression Results")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()