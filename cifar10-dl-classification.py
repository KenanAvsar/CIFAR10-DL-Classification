

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# select a random image
random_index = np.random.randint(0,len(X_train))
random_image = X_train[random_index]
random_label = y_train[random_index][0]


# display image
plt.imshow(random_image)
plt.axis('off')
plt.show()


random_image


random_image.shape



# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Randomly select 16 images from the training dataset
num_samples = 16
random_indices = np.random.choice(X_train.shape[0],num_samples,replace=False)
sample_images = X_train[random_indices]
sample_labels = y_train[random_indices]


plt.figure(figsize=(10,10))
for i in range (num_samples):
    plt.subplot(4,4,i+1)
    plt.imshow(sample_images[i])
    plt.title(class_names[int(sample_labels[i][0])])
    plt.axis('off')
plt.tight_layout()
plt.show()


# Normalization
X_train = X_train/255
X_test = X_test/255



num_samples = 16
random_indices = np.random.choice(X_train.shape[0],num_samples,replace=False)
sample_images = X_train[random_indices]
sample_labels = y_train[random_indices]



plt.figure(figsize=(10,10))
for i in range (num_samples):
    plt.subplot(4,4,i+1)
    plt.imshow(sample_images[i])
    plt.title(class_names[int(sample_labels[i][0])])
    plt.axis('off')
plt.tight_layout()
plt.show()



# specify the output layer is 10 classes
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# ## Modelling

model = Sequential()

# First Convolutional layer

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))

# Second Convolutional layer

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Fully connected layers

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='relu')) # Output layer with 10 neurons since there are 10 classes


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit(X_train, y_train, epochs = 10, batch_size = 64, validation_data=(X_test, y_test))


model.summary()



loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss :', loss)
print('Test Accuracy :', accuracy)


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ## Model 2

# see the difference using the ‘softmax’ activation code in the output layer.

model2 = Sequential()

# First Convolutional layer

model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model2.add(MaxPooling2D((2, 2)))

# Second Convolutional layer

model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))

# Fully connected layers

model2.add(Flatten())
model2.add(Dense(64, activation='relu'))
model2.add(Dense(10, activation='softmax'))



model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history2 = model2.fit(X_train, y_train, epochs = 10, batch_size = 64, validation_data=(X_test, y_test))


model2.summary()


loss2, accuracy2 = model2.evaluate(X_test,y_test)
print('Test Loss :', loss,'\nTest Accuracy :', accuracy)

print('Test Loss 2 : ', loss2, '\nTest Accuracy 2 :', accuracy2)



plt.plot(history2.history['loss'],label='Training Loss')
plt.plot(history2.history['val_loss'],label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



plt.plot(history2.history['accuracy'],label='Training Accuracy')
plt.plot(history2.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.plot(history2.history['accuracy'],label='Training Accuracy')
plt.plot(history2.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.legend()
plt.show()



plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label = 'Validation Loss')
plt.plot(history2.history['loss'],label='Training Loss')
plt.plot(history2.history['val_loss'],label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# ## Model 3

# change of the result when 1 more convolutional layer is added


model3 = Sequential()

# First Convolutional layer

model3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model3.add(MaxPooling2D((2, 2)))

# Second Convolutional layer

model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D((2, 2)))

# Third Convolutional layer

model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D((2, 2)))

# Fully connected layers

model3.add(Flatten())
model3.add(Dense(64, activation='relu'))
model3.add(Dense(10, activation='softmax'))


model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history3 = model3.fit(X_train, y_train, epochs=10,batch_size = 64, validation_data=(X_test, y_test))


model3.summary()


loss3, accuracy3 = model3.evaluate(X_test,y_test)
print('Test Loss :', loss,'\nTest Accuracy :', accuracy)

print('Test Loss 2 : ', loss2, '\nTest Accuracy 2 :', accuracy2)

print('Test Loss 3 : ', loss3, '\nTest Accuracy 3 :', accuracy3)


plt.plot(history2.history['accuracy'],label='model2 Training Accuracy')
plt.plot(history3.history['accuracy'],label='model3 Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.legend()
plt.show()


plt.plot(history2.history['val_accuracy'],label='model2 Validation Accuracy')
plt.plot(history3.history['val_accuracy'],label='model3 Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.legend()
plt.show()


plt.plot(history2.history['loss'],label='model2 Training Loss')
plt.plot(history3.history['loss'],label='model3 Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



plt.plot(history2.history['val_loss'],label='model2 Validation Loss')
plt.plot(history3.history['val_loss'],label='model3 Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# ## Model 4

# effect of optimiser=‘sgd’ change

model4 = Sequential()

# First Convolutional layer

model4.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model4.add(MaxPooling2D((2, 2)))

# Second Convolutional layer

model4.add(Conv2D(64, (3, 3), activation='relu'))
model4.add(MaxPooling2D((2, 2)))

# Third Convolutional layer

model4.add(Conv2D(64, (3, 3), activation='relu'))
model4.add(MaxPooling2D((2, 2)))

# Fully connected layers

model4.add(Flatten())
model4.add(Dense(64, activation='relu'))
model4.add(Dense(10, activation='softmax'))


model4.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])



history4 = model4.fit(X_train, y_train, epochs=10,batch_size = 64, validation_data=(X_test, y_test))


model4.summary()



loss4, accuracy4 = model3.evaluate(X_test,y_test)
print('Test Loss :', loss,'\nTest Accuracy :', accuracy)

print('Test Loss 2 : ', loss2, '\nTest Accuracy 2 :', accuracy2)

print('Test Loss 3 : ', loss3, '\nTest Accuracy 3 :', accuracy3)

print('Test Loss 4 : ', loss4, '\nTest Accuracy 4 :', accuracy4)


# ## Model 5

# effect of the number of epochs



model5 = Sequential()

# First Convolutional layer

model5.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model5.add(MaxPooling2D((2, 2)))

# Second Convolutional layer

model5.add(Conv2D(64, (3, 3), activation='relu'))
model5.add(MaxPooling2D((2, 2)))

# Third Convolutional layer

model5.add(Conv2D(64, (3, 3), activation='relu'))
model5.add(MaxPooling2D((2, 2)))

# Fully connected layers

model5.add(Flatten())
model5.add(Dense(64, activation='relu'))
model5.add(Dense(10, activation='softmax'))


model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history5 = model5.fit(X_train, y_train, epochs=50 ,batch_size = 64, validation_data=(X_test, y_test))


print('Test Loss 3:', loss3)
print('Test Accuracy 3:', accuracy3)

loss5, accuracy5 = model5.evaluate(X_test, y_test)
print('Test Loss 5:', loss5)
print('Test Accuracy 5:', accuracy5)

plt.plot(history3.history['accuracy'], label='Model3 Training Accuracy')
plt.plot(history3.history['val_accuracy'], label='Model3 Validation Accuracy')
plt.plot(history5.history['accuracy'], label='Model5 Training Accuracy')
plt.plot(history5.history['val_accuracy'], label='Model3 Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
     


# ## Model 6

# different spelling of model 2


model6 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=32,
                         kernel_size=3, # (3, 3)
                         activation="relu",
                         input_shape=(32, 32, 3)),

  tf.keras.layers.MaxPool2D(pool_size=2, # (2, 2)
                            padding="valid"), # padding can also be 'same'

  tf.keras.layers.Conv2D(64, 3, activation="relu"),


  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation="relu"),
  tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model6.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Fit the model
history6 = model6.fit(X_train,y_train,
                        epochs=10,
                        validation_data=(X_test, y_test))



