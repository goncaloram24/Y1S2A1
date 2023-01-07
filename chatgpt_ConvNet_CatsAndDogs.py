import tensorflow as tf
import numpy as np

EPOCHS = 20

# Load the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Select only the cat and dog classes
mask_train = np.array((y_train == 3) | (y_train == 5)).reshape(-1)
mask_test = np.array((y_test == 3) | (y_test == 5)).reshape(-1)
x_train = x_train[mask_train]
y_train = y_train[mask_train]
x_test = x_test[mask_test]
y_test = y_test[mask_test]

# Convert labels to binary (0 for cat, 1 for dog)
y_train = (y_train == 5).astype(int)
y_test = (y_test == 5).astype(int)

# Normalize the data
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Conv2D(32, (3, 3)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Conv2D(64, (3, 3)))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation("sigmoid"))

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_test,y_test))

# Evaluate the model
model.evaluate(x_test, y_test, verbose=2)

# Save the model
model.save("chatgpt_ConvNet_CatsAndDogs_models/model_epoch" + str(EPOCHS) + ".h5")
