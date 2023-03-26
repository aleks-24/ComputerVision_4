import tensorflow as tf

from keras.datasets import fashion_mnist

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold

from models import get_baseline_model, get_hyper_model,\
    get_model1, get_model2, get_model3, get_model4, get_kfold_model, \
    get_model_augmentation, get_learning_rate_model


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

# scheduler to change the learning rate
# every 5 epochs the learning rate is divided by 2
def scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        lr = lr / 2
    return lr

# callback for the fit function
callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)


# trains and eveluates the model
# only treins when it is not loaded beforehand
# if all_data is true, the model is trained on all the data
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
        model.fit(all_images,
                        all_labels,
                        epochs=15,
                        batch_size=128,
                        use_multiprocessing=True,
                        workers=6,
                        callbacks=callback)
    predictions = model.predict(test_images)
    show_statistics(test_labels, np.argmax(predictions, axis=1), model_name=model_name)


# use kfold cross validation to train on all the training data
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


# loads the model and evaluates it
def load_and_test_model(LoadModelFromDisk, model, model_name, callback=None, kfold=False, all_data=False):
    # needs to be done to save at the end
    if not LoadModelFromDisk:
        if kfold:
            kfold_train_and_evaluate_model(model, model_name=model_name)
        else:
            train_and_evaluate_model(model, model_name=model_name, callback=callback, load_model=False,
                                     all_data=all_data)
        model.save(model_name)
    else:
        train_and_evaluate_model(model, model_name=model_name, callback=callback, load_model=True,
                                 all_data=all_data)
    return model


# plots the history of the model
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


# show the metrics on the test set
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


# get and train all the models
def main():
    LoadModelFromDisk = False

    baseline_model = get_baseline_model(load_model=LoadModelFromDisk)
    load_and_test_model(LoadModelFromDisk, baseline_model, 'baseline_model_all_data', all_data=True)

    # model with smaller learning rate
    model1 = get_model1(load_model=LoadModelFromDisk)
    load_and_test_model(LoadModelFromDisk, model1, 'model1')

    # model with batch normalization
    model2 = get_model2(baseline_model, load_model=LoadModelFromDisk)
    load_and_test_model(LoadModelFromDisk, model2, 'model2')

    # model with one more dense layer
    model3 = get_model3(baseline_model, load_model=LoadModelFromDisk)
    load_and_test_model(LoadModelFromDisk, model3, 'model3')

    # baseline model with dropout
    model4 = get_model4(baseline_model, load_model=LoadModelFromDisk)
    load_and_test_model(LoadModelFromDisk, model4, 'model4_all_data', all_data=True)
    #
    # baseline model with reducing learning rate
    model_learning_rate = get_learning_rate_model(load_model=LoadModelFromDisk)
    load_and_test_model(LoadModelFromDisk, model_learning_rate, 'model_learning_rate', callback=callback)

    # model with data augmentation
    model_augmentation = get_model_augmentation(baseline_model, load_model=LoadModelFromDisk)
    load_and_test_model(LoadModelFromDisk, model_augmentation, 'model_augmentation')

    # baseline model with kfold cross validation
    kfold_model = get_kfold_model(load_model=LoadModelFromDisk)
    load_and_test_model(LoadModelFromDisk, kfold_model, 'model_kfold', kfold=True)

    # # baseline model with hyperparameter tuning for learning rate
    model_hyper = get_hyper_model(load_model=LoadModelFromDisk)
    load_and_test_model(LoadModelFromDisk, model_hyper, 'model hyperparameter tuning')





if __name__ == '__main__':
    main()
