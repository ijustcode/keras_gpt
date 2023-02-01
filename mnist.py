from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical

# load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

val_size = int(len(X_test) * 0.2)
X_val = X_test[:val_size]
y_val = y_test[:val_size]
X_test = X_test[val_size:]
y_test = y_test[val_size:]


# # reshape the input data to be 28x28x1 (height x width x channels)
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# normalize the input data to be between 0 and 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# # convert the output labels to categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# create a sequential model
model = Sequential()

# add the convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# add the dense layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

import random

# Choose 20 random indices from the training set
indices = random.sample(range(len(X_train)), 20)

# Plot the images and labels in a 4x5 grid
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.ravel()
for i, ax in enumerate(axes):
    ax.imshow(X_train[indices[i]])
    ax.set_title(str(y_train[indices[i]]))
    ax.axis('off')

plt.tight_layout()
plt.show()
