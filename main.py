import tensorflow as tf

from keras.datasets import fashion_mnist
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load the data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images, c_test = train_images / 255.0, test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# split train in validation and train set

val_images = train_images[:12000]
val_labels = train_labels[:12000]
train_images = train_images[48000:]
train_labels = train_labels[48000:]


def get_baseline_model():
    # Build the model
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=8,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='valid',
                                     activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(filters=16,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='valid',
                                     activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=32,
                                     kernel_size=(5, 5),
                                     strides=(1, 1),
                                     padding='valid',
                                     activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    print(model.summary())
    return model


baseline_model = get_baseline_model()

baseline_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = baseline_model.fit(train_images, train_labels, epochs=15,
                    validation_data=(val_images, val_labels), batch_size= 32)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = baseline_model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
def main():
    print('Hello World')

if __name__ == '__main__':
    main()
