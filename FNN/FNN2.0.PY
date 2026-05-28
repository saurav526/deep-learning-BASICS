import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0


model = Sequential()


model.add(Flatten(input_shape=(28, 28)))


model.add(Dense(128, activation='relu'))


model.add(Dense(64, activation='relu'))


model.add(Dense(10, activation='softmax'))


model.summary()


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


model.fit( x_train, y_train,epochs=5,
    validation_data=(x_test, y_test)
)


test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_acc)


predictions = model.predict(x_test)

print("Predicted Digit:", predictions[0].argmax())
print("Actual Digit:", y_test[0])