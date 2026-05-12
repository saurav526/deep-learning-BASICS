import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

X = []
y = []

for i in range(1, 100):
    X.append([i, i+1, i+2])
    y.append(i+3)

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential()

model.add(SimpleRNN(
    units=50,
    activation='tanh',
    input_shape=(3,1)
))

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(X, y, epochs=200, verbose=1)

test = np.array([[100,101,102]])
test = test.reshape((1,3,1))

prediction = model.predict(test)

print(prediction[0][0])