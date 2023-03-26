import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from keras.datasets import fashion_mnist
from keras import layers

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold

# Load the data
(all_images, all_labels), (test_images, test_labels) = fashion_mnist.load_data()

all_images, test_images = all_images / 255.0, test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(all_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[all_labels[i]])
plt.savefig('images/fashion_mnist.png')
plt.show()

# split train in validation and train set

val_images = all_images[:12000]
val_labels = all_labels[:12000]
train_images = all_images[48000:]
train_labels = all_labels[48000:]


def get_baseline_model(load_model=False):
    # Build the model
    if not load_model:
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.Dense(10, activation='softmax'))
        print(model.summary())

        compile_model(model)
    else:
        model = tf.keras.models.load_model('baseline_model')
    return model



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


def get_model4(baseline_model, load_model=False):
    if not load_model:
        model4 = insert_layer_after(baseline_model, 6, layers.Dropout(0.5))
        compile_model(model4)
    else:
        model4 = tf.keras.models.load_model('model4')
    print(model4.summary())

    return model4


def get_hyper_model(load_model=False):
    if not load_model:
        tuner = kt.Hyperband(model_builder,
                             objective='val_accuracy',
                             max_epochs=15,
                             factor=3,
                             directory='hyperband',
                             project_name='ComputerVision4')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(all_images, all_labels, epochs=15, validation_split=0.2, callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete. The optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
        """)
        model_hyper = tuner.hypermodel.build(best_hps)
        model_hyper.save('model_hyper')

    else:
        model_hyper = tf.keras.models.load_model('model_hyper')
    print(model_hyper.summary())
    return model_hyper

def model_builder(hp):
    model_kfold = get_baseline_model(load_model=False)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model_kfold.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model_kfold




def get_model_augmentation(baseline_model, load_model=False):
    # augment the training data
    if not load_model:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal",
                              input_shape=(28, 28, 1)),
            layers.GaussianNoise(0.1),
            layers.RandomContrast(0.1)
        ])
        model_augmentation = insert_layer_after(baseline_model, 0, data_augmentation)
        compile_model(model_augmentation)
    else:
        model_augmentation = tf.keras.models.load_model('model_augmentation')
    print(model_augmentation.summary())
    return model_augmentation

def scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        lr = lr / 2
    return lr


callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)


def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])



def train_and_evaluate_model(model, model_name='model', callback=None, load_model=False, all_data=False):
    if not load_model and not all_data:
        history = model.fit(train_images,
                        train_labels,
                        epochs=15,
                        validation_data=(val_images, val_labels),
                        batch_size=128,
                        use_multiprocessing=True,
                        workers=6,
                        callbacks=callback)


        plot_history(history, 'accuracy', y_limit=[0, 1], model_name=model_name)
        plot_history(history, 'loss', y_limit=[0, 1], model_name=model_name)

    if all_data:
        history = model.fit(all_images,
                        all_labels,
                        epochs=15,
                        batch_size=128,
                        use_multiprocessing=True,
                        workers=6,
                        callbacks=callback)
    predictions = model.predict(test_images)
    show_statistics(test_labels, np.argmax(predictions, axis=1), model_name=model_name)


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
                            epochs=15,
                            validation_data=(fold_images[test], fold_labels[test]),
                            batch_size=128,
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
    plt.savefig(f'images/{model_name}_{metric}.png')
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
    ConfusionMatrixDisplay.from_predictions(true_label, pred_label, display_labels=class_names, xticks_rotation=25)
    plt.tight_layout()
    plt.savefig(f'images/{model_name}_confusion_matrix.png')
    plt.show()


def get_model1(load_model):
    if not load_model:
        model1 = get_baseline_model()
        model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.3e-3),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
    else:
        model1 = tf.keras.models.load_model('model1')
    print(model1.summary())
    return model1


def get_model2(model, load_model):
    if not load_model:
        model2 = get_baseline_model()
        model2 = insert_layer_after(model2, 1, tf.keras.layers.BatchNormalization())
        compile_model(model2)
    else:
        model2 = tf.keras.models.load_model('model2')
    print(model2.summary())
    return model2


def get_model3(model, load_model):
    if not load_model:
        model3 = replace_intermediate_layer(model, 5,
                                            tf.keras.layers.Dense(128,
                                                                  activation='relu',
                                                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        compile_model(model3)
    else:
        model3 = tf.keras.models.load_model('model3')
    print(model3.summary())
    return model3


def main():
    LoadModelFromDisk = False
    #
    #baseline_model = get_baseline_model(load_model=LoadModelFromDisk)
    #load_and_test_model(LoadModelFromDisk, baseline_model, 'baseline_model_all_data', all_data=True)
    #
    # # model with smaller learning rate
    # model1 = get_model1(load_model=LoadModelFromDisk)
    # load_and_test_model(LoadModelFromDisk, model1, 'model1')
    #
    # # model with batch normalization
    # model2 = get_model2(baseline_model, load_model=LoadModelFromDisk)
    # load_and_test_model(LoadModelFromDisk, model2, 'model2')
    #
    # # model with one more dense layer
    # model3 = get_model3(baseline_model, load_model=LoadModelFromDisk)
    # load_and_test_model(LoadModelFromDisk, model3, 'model3')

    # baseline model with dropout
    # model4 = get_model4(baseline_model, load_model=LoadModelFromDisk)
    # load_and_test_model(LoadModelFromDisk, model4, 'model4_all_data', all_data=True)
    # #
    # # baseline model with reducing learning rate
    # model_learning_rate = get_baseline_model(load_model=LoadModelFromDisk)
    # load_and_test_model(LoadModelFromDisk, model_learning_rate, 'model_learning_rate', callback=callback)
    #
    # # model with data augmentation
    # model_augmentation = get_model_augmentation(baseline_model, load_model=LoadModelFromDisk)
    # load_and_test_model(LoadModelFromDisk, model_augmentation, 'model_augmentation')

    # # baseline model with kfold cross validation
    model_hyper = get_hyper_model(load_model=LoadModelFromDisk)
    load_and_test_model(LoadModelFromDisk, model_hyper, 'model hyperparameter tuning')


def load_and_test_model(LoadModelFromDisk, model, model_name, callback=None, kfold=False, all_data=False):
    if not LoadModelFromDisk:
        if kfold:
            kfold_train_and_evaluate_model(model, model_name=model_name)
        else:
            train_and_evaluate_model(model, model_name=model_name, callback=callback, load_model=False, all_data=all_data)
        model.save(model_name)
    else:
        train_and_evaluate_model(model, model_name=model_name, callback=callback, load_model=True, all_data=all_data)
    return model


if __name__ == '__main__':
    main()
