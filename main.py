import tensorflow as tf

from keras.datasets import fashion_mnist
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold

# Load the data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images, c_test = train_images / 255.0, test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
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
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=16,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='valid',
                            activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=32,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='valid',
                            activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    print(model.summary())

    compile_model(model)
    return model


def compile_model(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])


# inserting layer according to:
# https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
def insert_layer_after(model, layer_id, new_layer):
    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)

    new_model = tf.keras.Model(inputs=layers[0].input, outputs=x)
    return new_model


# replacing a layer according to:
# https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
def replace_intermediate_layer(model, layer_id, new_layer):
    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = tf.keras.Model(inputs=layers[0].input, outputs=x)
    return new_model


def get_model4(baseline_model):
    model4 = replace_intermediate_layer(
        baseline_model,
        0,
        layers.Conv2D(filters=16,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='valid',
                      activation='relu',
                      input_shape=(28, 28, 1)))
    print(model4.summary())
    compile_model(model4)
    return model4


def get_model3(model):
    model3 = insert_layer_after(model, 8, layers.Dropout(0.5))
    print(model3.summary())
    compile_model(model3)
    return model3


def get_model2(baseline_model):
    x2 = tf.keras.layers.Flatten()(baseline_model.layers[-4].output)
    output = tf.keras.layers.Dense(128, activation='relu')(x2)
    output = tf.keras.layers.Dense(10, activation='softmax')(output)
    model2 = tf.keras.Model(inputs=baseline_model.input, outputs=output)
    print(model2.summary())
    compile_model(model2)
    return model2


def get_model1(baseline_model):
    # augment the training data
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(28,
                                                                  28,
                                                                  1)),
        layers.GaussianNoise(0.1),
        layers.experimental.preprocessing.RandomContrast(0.1)
    ])
    model1 = insert_layer_after(baseline_model, 0, data_augmentation)
    print(model1.summary())
    compile_model(model1)
    return model1


def train_and_evaluate_model(model, model_name='model'):


    history = model.fit(train_images,
                        train_labels,
                        epochs=10,
                        validation_data=(val_images, val_labels),
                        batch_size=32,
                        use_multiprocessing=True,
                        workers=6)
    plot_history(history, 'accuracy', y_limit=[0, 1], model_name=model_name)
    plot_history(history, 'loss', y_limit=[0, 1], model_name=model_name)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    predictions = model.predict(test_images)
    show_statistics(test_labels, np.argmax(predictions, axis=1), model_name=model_name)
    print(f"test acuracy: {test_acc}, test loss: {test_loss}")


def kfold_train_and_evaluate_model(model, model_name='model'):
    # use kfold to train and evaluate the model
    kfold = KFold(n_splits=5, shuffle=True)
    average_scores = []
    fold_no = 1
    fold_images = np.concatenate((train_images, val_images))
    fold_labels = np.concatenate((train_labels, val_labels))
    for train, test in kfold.split(fold_images, fold_labels):
        print(f"fold: {fold_no}")
        history = model.fit(fold_images[train],
                            fold_labels[train],
                            epochs=10,
                            validation_data=(fold_images[test], fold_labels[test]),
                            batch_size=32,
                            use_multiprocessing=True,
                            workers=6)
        plot_history(history, 'accuracy', y_limit=[0, 1], model_name=model_name)
        plot_history(history, 'loss', y_limit=[0, 1], model_name=model_name)
        test_loss, test_acc = model.evaluate(test_images,test_labels, verbose=1)
        predictions = model.predict(test_images)
        show_statistics(test_labels, np.argmax(predictions, axis=1), model_name=model_name)
        print(f"test accuracy for fold: {fold_no}: {test_acc}, test loss: {test_loss}")
        average_scores.append(test_acc)
        fold_no = fold_no + 1

    print(f"Average test accuracy: {np.mean(average_scores)}")

def plot_history(history, metric='accuracy', y_limit=[0.5, 1], model_name='model'):
    plt.plot(history.history[metric], label=metric)
    plt.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
    plt.title(f'{model_name}: {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.ylim(y_limit)
    plt.legend(loc='lower left')
    plt.show()


def show_statistics(true_label, pred_label, model_name='model'):
    # print statistics and plot confusion matrix
    print(model_name)
    print("------------------------------------------")
    print(classification_report(true_label, pred_label))
    print("Accuracy: " + str(accuracy_score(true_label, pred_label)))
    print("Precision: " + str(precision_score(true_label, pred_label, average='macro', zero_division=0)))
    print("Recall: " + str(recall_score(true_label, pred_label, average='macro')))
    print("F1: " + str(f1_score(true_label, pred_label, average='macro')))
    print("------------------------------------------")

    ConfusionMatrixDisplay(confusion_matrix(true_label, pred_label)).plot()
    plt.show()


def main():

    baseline_model = get_baseline_model()

    #train_and_evaluate_model(baseline_model, model_name='baseline_model')

    # baseline model with 1 extra layer in the fully connected layers
    model1 = get_model1(baseline_model)

    kfold_train_and_evaluate_model(model1, model_name='model1')

    # baseline model without the last pooling layer
    model2 = get_model2(baseline_model)
    train_and_evaluate_model(model2, model_name='model2')

    # baseline model with a dropout layer
    model3 = get_model3(baseline_model)
    train_and_evaluate_model(model3, model_name='model3')

    model4 = get_model4(baseline_model)
    train_and_evaluate_model(model4, model_name='model4')


if __name__ == '__main__':
    main()
