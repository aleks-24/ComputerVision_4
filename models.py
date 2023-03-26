import keras_tuner as kt
import tensorflow as tf
from keras import layers

from main import all_images, all_labels


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


# compiles a model
def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])


# the baseline model
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



# model 1 with smaller learning rate
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


# model 2 with batch normalization
def get_model2(model, load_model):
    if not load_model:
        model2 = get_baseline_model()
        model2 = insert_layer_after(model2, 1, tf.keras.layers.BatchNormalization())
        compile_model(model2)
    else:
        model2 = tf.keras.models.load_model('model2')
    print(model2.summary())
    return model2


# model 3 with regularization
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


# model 4 with dropout
def get_model4(baseline_model, load_model=False):
    if not load_model:
        model4 = insert_layer_after(baseline_model, 6, layers.Dropout(0.5))
        compile_model(model4)
    else:
        model4 = tf.keras.models.load_model('model4')
    print(model4.summary())

    return model4


# model with hyper parameter tuning for learning rate
# according to https://www.tensorflow.org/tutorials/keras/keras_tuner
def get_hyper_model(load_model=False):
    if not load_model:
        tuner = kt.Hyperband(model_builder,
                             objective='val_accuracy',
                             max_epochs=15,
                             factor=3,
                             directory='hyperband',
                             project_name='ComputerVision4')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(all_images, all_labels, epochs=15, validation_split=0.2, callbacks=[stop_early], use_multiprocessing=True,
                        workers= 6)

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


# model builder for hyper parameter tuning
def model_builder(hp):
    model= get_baseline_model(load_model=False)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



# model with data augmentation
def get_model_augmentation(baseline_model, load_model=False):
    # augment the training data
    if not load_model:
        model_augmentation = tf.keras.Sequential()
        model_augmentation.add(layers.RandomFlip("horizontal", input_shape=(28, 28, 1)))
        model_augmentation.add(layers.GaussianNoise(0.1))
        model_augmentation.add(layers.RandomContrast(0.1))
        model_augmentation.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform',
                                input_shape=(28, 28, 1)))
        model_augmentation.add(layers.MaxPooling2D((2, 2)))
        model_augmentation.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform',
                                input_shape=(28, 28, 1)))
        model_augmentation.add(layers.MaxPooling2D((2, 2)))
        model_augmentation.add(layers.Flatten())
        model_augmentation.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model_augmentation.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model_augmentation.add(layers.Dense(10, activation='softmax'))
        print(model_augmentation.summary())

        compile_model(model_augmentation)
    else:
        model_augmentation = tf.keras.models.load_model('model_augmentation')
    print(model_augmentation.summary())
    return model_augmentation



# model with k-fold cross validation
def get_kfold_model(load_model):
    if not load_model:
        model_kfold = get_baseline_model()
        compile_model(model_kfold)
    else:
        model_kfold = tf.keras.models.load_model('model_kfold')
    print(model_kfold.summary())
    return model_kfold


# model with learning rate scheduler
def get_learning_rate_model(load_model):
    if not load_model:
        model_learning_rate = get_baseline_model()
        compile_model(model_learning_rate)
    else:
        model_learning_rate = tf.keras.models.load_model('model_learning_rate')
    print(model_learning_rate.summary())
    return model_learning_rate



