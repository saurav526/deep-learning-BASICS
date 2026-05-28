import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

# Build CNN model
model = models.Sequential()

# Convolution Layer
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(28, 28, 1)))

# Pooling Layer
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolution Layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Second Pooling Layer
model.add(layers.MaxPooling2D((2, 2)))

# Flatten Layer
model.add(layers.Flatten())

# Fully Connected Layer
model.add(layers.Dense(64, activation='relu'))

# Output Layer
model.add(layers.Dense(10, activation='softmax'))

# Model Summary
model.summary()

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_acc)

# Prediction
predictions = model.predict(x_test)

print("Predicted Digit:", predictions[0].argmax())
print("Actual Digit:", y_test[0])

# Display image
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.show()
