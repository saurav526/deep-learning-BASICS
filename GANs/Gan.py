import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# Hyperparameters
# ==========================================
BATCH_SIZE = 128
NOISE_DIM = 100
EPOCHS = 50
LEARNING_RATE = 0.0002

# ==========================================
# Load and preprocess MNIST
# ==========================================
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32")
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)

dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(60000).batch(BATCH_SIZE)

# ==========================================
# Generator
# ==========================================
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, input_shape=(NOISE_DIM,)),
        layers.LeakyReLU(0.2),

        layers.Dense(512),
        layers.LeakyReLU(0.2),

        layers.Dense(1024),
        layers.LeakyReLU(0.2),

        layers.Dense(28 * 28, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])

    return model

# ==========================================
# Discriminator
# ==========================================
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),

        layers.Dense(1024),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Dense(512),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Dense(256),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Dense(1, activation='sigmoid')
    ])

    return model

generator = build_generator()
discriminator = build_discriminator()

# ==========================================
# Loss Functions
# ==========================================
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = cross_entropy(
        tf.zeros_like(fake_output),
        fake_output
    )

    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(
        tf.ones_like(fake_output),
        fake_output
    )

# ==========================================
# Optimizers
# ==========================================
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.5
)

discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.5
)

# ==========================================
# Training Step
# ==========================================
@tf.function
def train_step(images):

    batch_size = tf.shape(images)[0]

    noise = tf.random.normal([batch_size, NOISE_DIM])

    with tf.GradientTape() as gen_tape, \
         tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(
            generated_images,
            training=True
        )

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(
            real_output,
            fake_output
        )

    gradients_of_generator = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables
    )

    gradients_of_discriminator = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(
            gradients_of_generator,
            generator.trainable_variables
        )
    )

    discriminator_optimizer.apply_gradients(
        zip(
            gradients_of_discriminator,
            discriminator.trainable_variables
        )
    )

    return gen_loss, disc_loss

# ==========================================
# Save Generated Images
# ==========================================
os.makedirs("generated_images", exist_ok=True)

seed = tf.random.normal([16, NOISE_DIM])

def save_images(epoch):

    predictions = generator(seed, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)

        img = predictions[i, :, :, 0]
        img = (img + 1) / 2.0

        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.savefig(
        f"generated_images/epoch_{epoch:03d}.png"
    )

    plt.close()

# ==========================================
# Training Loop
# ==========================================
for epoch in range(EPOCHS):

    for image_batch in dataset:
        g_loss, d_loss = train_step(image_batch)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"G Loss: {g_loss:.4f} | "
        f"D Loss: {d_loss:.4f}"
    )

    save_images(epoch + 1)

# ==========================================
# Save Models
# ==========================================
generator.save("generator.keras")
discriminator.save("discriminator.keras")

print("Training Complete!")