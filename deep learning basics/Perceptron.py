import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

weights = np.random.randn(2)
bias = np.random.randn()


def activation(x):
    return 1 if x >= 0 else 0


learning_rate = 0.1

for epoch in range(10):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        y_pred = activation(linear_output)
        
        error = y[i] - y_pred
        
        
        weights += learning_rate * error * X[i]
        bias += learning_rate * error


plt.figure()


for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1])
    else:
        plt.scatter(X[i][0], X[i][1])

# Decision boundary: w1*x1 + w2*x2 + b = 0
x_vals = np.linspace(-0.5, 1.5, 100)
y_vals = -(weights[0]*x_vals + bias) / weights[1]

plt.plot(x_vals, y_vals)

plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("Perceptron Decision Boundary (AND Logic)")

plt.show()