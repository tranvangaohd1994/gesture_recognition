.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=lr_schedule(0), decay=1e-5, momentum=0.9),
              metrics=['accuracy'])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

filepath = os.path.join(save_dir, model_name)


lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_scheduler,lr_reducer]


datagen = ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of dataset
    featurewise_std_normalization=False,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # epsilon for ZCA whitening
    zca_epsilon=1e-06,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # set range for random shear
    shear_range=0.,
    # set range for random zoom
    zoom_range=0.,
    # set range for random channel shifts
    channel_shift_range=0.,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # value used for fill_mode = "constant"
    cval=0.,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=False,
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0
    )


model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                         validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        steps_per_epoch = len(x_train)//batch_size,
                        callbacks=callbacks)
scores = model.evaluate(x_test, y_test, verbose=1)

Epoch 1/200
2020-07-16 11:55:45.579274: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcublas.so.10.0
2020-07-16 11:55:45.712483: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
31/31 [==============================] - 6s 199ms/step - loss: 61.0374 - accuracy: 0.2603 - val_loss: 58.1093 - val_accuracy: 0.1309
Epoch 2/200
31/31 [==============================] - 3s 94ms/step - loss: 54.3552 - accuracy: 0.4013 - val_loss: 51.2305 - val_accuracy: 0.2321
Epoch 3/200
31/31 [==============================] - 3s 99ms/step - loss: 47.9503 - accuracy: 0.5060 - val_loss: 45.4614 - val_accuracy: 0.1955
Epoch 4/200
31/31 [==============================] - 3s 95ms/step - loss: 42.2219 - accuracy: 0.6208 - val_loss: 40.4835 - val_accuracy: 0.2164
Epoch 5/200
31/31 [==============================] - 3s 95ms/step - loss: 37.0443 - accuracy: 0.7317 - val_loss: 36.1821 - val_accuracy: 0.1326
Epoch 6/200
31/31 [==============================] - 3s 95ms/step - loss: 32.5234 - accuracy: 0.7779 - val_loss: 32.5856 - val_accuracy: 0.1187
Epoch 7/200
31/31 [==============================] - 3s 95ms/step - loss: 28.5925 - accuracy: 0.8177 - val_loss: 29.4391 - val_accuracy: 0.1117
Epoch 8/200
31/31 [==============================] - 3s 99ms/step - loss: 25.1530 - accuracy: 0.8595 - val_loss: 26.6698 - val_accuracy: 0.1117
Epoch 9/200
31/31 [==============================] - 3s 97ms/step - loss: 22.1478 - accuracy: 0.8874 - val_loss: 24.1873 - val_accuracy: 0.1117
