### Assignment 3 Results


#### 0. Number of Parameters 
##### 0.a For base network  
* 1,172,410

##### 0.b For my network  
* 84,571  
  
#### 1. Final Validation Accuracy 
##### 1.a For base network  
* 82.1

##### 1.b For my network  
* 86.1  
  
#### 2. My model definition  
import keras

d = 0.1
f = 60
dm = 3
isDropout = True
#target val_acc = 82.6
if RAJA == True: # Developed by Raja 
    # Define the model
    model2 = Sequential()
    #filters become output channels (filter count of 1x1)
    #depth_multiplier may be assumed as the channel of kernels (in legacy 3x3 conv)
    #filters = 3x3x1 filter count
    #depth_multiplier = 1x1 filter count

    layer1f = f #layer 1 filter count
    model2.add(SeparableConv2D(filters=layer1f, kernel_size=(3, 3), depth_multiplier=dm, input_shape=(32, 32, 3), use_bias=False, name="FirstDwsConv"))#30
    #RF=3
    model2.add(BatchNormalization())
    model2.add(Activation('relu'))
    if isDropout:
      model2.add(Dropout(d))
    layer2f = f #layer 2 filter count
    model2.add(SeparableConv2D(filters=layer2f, kernel_size=3, depth_multiplier=dm, use_bias=False, name="SecondDwsConv"))#28
    #RF=5
    model2.add(BatchNormalization())
    model2.add(Activation('relu'))
    if isDropout:
      model2.add(Dropout(d))
    layer3f = f #layer 3 filter count
    model2.add(SeparableConv2D(filters=layer3f, kernel_size=3, depth_multiplier=dm, use_bias=False, name="ThirdDwsConv"))#26
    #RF=7
    model2.add(BatchNormalization())
    model2.add(Activation('relu'))
    if isDropout:
      model2.add(Dropout(d))
    
    model2.add(MaxPooling2D(pool_size=(2, 2)))#13
    #RF=8
    model2.add(SeparableConv2D(10, 1, 1, depth_multiplier=dm, use_bias=False, name="PointwiseDwsConv"))#13
    #RF=8

    layer4f = f #layer 4 filter count
    model2.add(SeparableConv2D(filters=layer4f, kernel_size=(3, 3), depth_multiplier=dm, use_bias=False, padding='same', name="FourthDwsConv"))#13
    #RF=12
    model2.add(BatchNormalization())
    model2.add(Activation('relu'))
    if isDropout:
      model2.add(Dropout(d))
    layer5f = f #layer 5 filter count
    model2.add(SeparableConv2D(filters=layer5f, kernel_size=(3, 3), depth_multiplier=dm, use_bias=False, padding='same', name="FifthDwsConv"))#13
    #RF=16
    model2.add(BatchNormalization())
    model2.add(Activation('relu'))
    if isDropout:
      model2.add(Dropout(d))
    
    layer6f = f #layer 6 filter count
    model2.add(SeparableConv2D(filters=layer6f, kernel_size=(3, 3), depth_multiplier=dm, use_bias=False, padding='same', name="SixthDwsConv"))#13
    #RF=20
    model2.add(BatchNormalization())
    model2.add(Activation('relu'))
    if isDropout:
      model2.add(Dropout(d))
    layer7f = f #layer 7 filter count
    model2.add(SeparableConv2D(filters=layer7f, kernel_size=(3, 3), depth_multiplier=dm, use_bias=False, name="SeventhDwsConv"))#11
    #RF=24
    model2.add(BatchNormalization())
    model2.add(Activation('relu'))
    if isDropout:
      model2.add(Dropout(d))
    layer8f = f #layer 8 filter count
    model2.add(SeparableConv2D(filters=layer8f, kernel_size=(3, 3), depth_multiplier=dm, use_bias=False, name="EighthDwsConv"))#7
    #RF=28
    model2.add(BatchNormalization())
    
    model2.add(SeparableConv2D(filters=10, kernel_size=(3, 3), depth_multiplier=dm, use_bias=False, name="NinethDwsConv"))#5
    #RF=32
    model2.add(BatchNormalization())

    #model2.add(Flatten())
    model2.add(GlobalAveragePooling2D())
    model2.add(Activation('softmax'))

    # Compile the model
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model2.summary()
    # train the model
    start = time.time()
    # Train the model
    datagen2 = ImageDataGenerator(zoom_range=0.0, 
                            horizontal_flip=True)
    snapshot_filepath = "/opt/saved_weight_file.hdf5"
    cp_callback = keras.callbacks.ModelCheckpoint(snapshot_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
    CALLBACKS = [cp_callback, lr_scheduler]
    model_info = model2.fit_generator(datagen2.flow(train_features, train_labels, batch_size = 128),
                                    samples_per_epoch = train_features.shape[0], nb_epoch = 50, 
                                    validation_data = (test_features, test_labels), verbose=1,
                                    callbacks = CALLBACKS)
    end = time.time()
    print ("Model took %0.2f seconds to train"%(end - start))
    # plot model history
    plot_model_history(model_info)
    # compute test accuracy
    model2.load_weights(snapshot_filepath)
    print ("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model2))



#### 3. My 50 epochs log  

Epoch 1/50
390/390 [==============================] - 39s 99ms/step - loss: 1.4610 - acc: 0.5001 - val_loss: 1.3249 - val_acc: 0.5454

Epoch 00001: val_acc improved from -inf to 0.54540, saving model to /opt/saved_weight_file.hdf5
Epoch 2/50
390/390 [==============================] - 33s 86ms/step - loss: 1.1031 - acc: 0.6340 - val_loss: 1.0807 - val_acc: 0.6313

Epoch 00002: val_acc improved from 0.54540 to 0.63130, saving model to /opt/saved_weight_file.hdf5
Epoch 3/50
390/390 [==============================] - 34s 86ms/step - loss: 0.9574 - acc: 0.6819 - val_loss: 1.0831 - val_acc: 0.6303

Epoch 00003: val_acc did not improve from 0.63130
Epoch 4/50
390/390 [==============================] - 33s 86ms/step - loss: 0.8618 - acc: 0.7154 - val_loss: 0.9370 - val_acc: 0.6784

Epoch 00004: val_acc improved from 0.63130 to 0.67840, saving model to /opt/saved_weight_file.hdf5
Epoch 5/50
390/390 [==============================] - 33s 86ms/step - loss: 0.7854 - acc: 0.7372 - val_loss: 0.8947 - val_acc: 0.7010

Epoch 00005: val_acc improved from 0.67840 to 0.70100, saving model to /opt/saved_weight_file.hdf5
Epoch 6/50
390/390 [==============================] - 33s 86ms/step - loss: 0.7386 - acc: 0.7532 - val_loss: 0.7479 - val_acc: 0.7477

Epoch 00006: val_acc improved from 0.70100 to 0.74770, saving model to /opt/saved_weight_file.hdf5
Epoch 7/50
390/390 [==============================] - 33s 85ms/step - loss: 0.6950 - acc: 0.7668 - val_loss: 0.7327 - val_acc: 0.7485

Epoch 00007: val_acc improved from 0.74770 to 0.74850, saving model to /opt/saved_weight_file.hdf5
Epoch 8/50
390/390 [==============================] - 33s 86ms/step - loss: 0.6602 - acc: 0.7780 - val_loss: 0.7288 - val_acc: 0.7476

Epoch 00008: val_acc did not improve from 0.74850
Epoch 9/50
390/390 [==============================] - 33s 86ms/step - loss: 0.6304 - acc: 0.7873 - val_loss: 0.7397 - val_acc: 0.7450

Epoch 00009: val_acc did not improve from 0.74850
Epoch 10/50
390/390 [==============================] - 33s 86ms/step - loss: 0.6043 - acc: 0.7972 - val_loss: 0.6921 - val_acc: 0.7652

Epoch 00010: val_acc improved from 0.74850 to 0.76520, saving model to /opt/saved_weight_file.hdf5
Epoch 11/50
390/390 [==============================] - 33s 86ms/step - loss: 0.5864 - acc: 0.8003 - val_loss: 0.6269 - val_acc: 0.7869

Epoch 00011: val_acc improved from 0.76520 to 0.78690, saving model to /opt/saved_weight_file.hdf5
Epoch 12/50
390/390 [==============================] - 33s 86ms/step - loss: 0.5740 - acc: 0.8048 - val_loss: 0.7765 - val_acc: 0.7331

Epoch 00012: val_acc did not improve from 0.78690
Epoch 13/50
390/390 [==============================] - 33s 86ms/step - loss: 0.5491 - acc: 0.8119 - val_loss: 0.7479 - val_acc: 0.7487

Epoch 00013: val_acc did not improve from 0.78690
Epoch 14/50
390/390 [==============================] - 33s 86ms/step - loss: 0.5371 - acc: 0.8175 - val_loss: 0.6095 - val_acc: 0.7859

Epoch 00014: val_acc did not improve from 0.78690
Epoch 15/50
390/390 [==============================] - 33s 86ms/step - loss: 0.5225 - acc: 0.8225 - val_loss: 0.5798 - val_acc: 0.8060

Epoch 00015: val_acc improved from 0.78690 to 0.80600, saving model to /opt/saved_weight_file.hdf5
Epoch 16/50
390/390 [==============================] - 33s 85ms/step - loss: 0.5108 - acc: 0.8259 - val_loss: 0.6084 - val_acc: 0.7940

Epoch 00016: val_acc did not improve from 0.80600
Epoch 17/50
390/390 [==============================] - 33s 86ms/step - loss: 0.5052 - acc: 0.8298 - val_loss: 0.5828 - val_acc: 0.8029

Epoch 00017: val_acc did not improve from 0.80600
Epoch 18/50
390/390 [==============================] - 33s 86ms/step - loss: 0.4931 - acc: 0.8326 - val_loss: 0.5982 - val_acc: 0.7919

Epoch 00018: val_acc did not improve from 0.80600
Epoch 19/50
390/390 [==============================] - 33s 86ms/step - loss: 0.4780 - acc: 0.8370 - val_loss: 0.5349 - val_acc: 0.8163

Epoch 00019: val_acc improved from 0.80600 to 0.81630, saving model to /opt/saved_weight_file.hdf5
Epoch 20/50
390/390 [==============================] - 33s 85ms/step - loss: 0.4741 - acc: 0.8383 - val_loss: 0.5893 - val_acc: 0.8009

Epoch 00020: val_acc did not improve from 0.81630
Epoch 21/50
390/390 [==============================] - 33s 86ms/step - loss: 0.4680 - acc: 0.8408 - val_loss: 0.5555 - val_acc: 0.8107

Epoch 00021: val_acc did not improve from 0.81630
Epoch 22/50
390/390 [==============================] - 33s 86ms/step - loss: 0.4537 - acc: 0.8441 - val_loss: 0.5998 - val_acc: 0.8016

Epoch 00022: val_acc did not improve from 0.81630
Epoch 23/50
390/390 [==============================] - 33s 86ms/step - loss: 0.4489 - acc: 0.8483 - val_loss: 0.5434 - val_acc: 0.8169

Epoch 00023: val_acc improved from 0.81630 to 0.81690, saving model to /opt/saved_weight_file.hdf5
Epoch 24/50
390/390 [==============================] - 33s 86ms/step - loss: 0.4410 - acc: 0.8504 - val_loss: 0.5251 - val_acc: 0.8244

Epoch 00024: val_acc improved from 0.81690 to 0.82440, saving model to /opt/saved_weight_file.hdf5
Epoch 25/50
390/390 [==============================] - 33s 86ms/step - loss: 0.4378 - acc: 0.8513 - val_loss: 0.5196 - val_acc: 0.8201

Epoch 00025: val_acc did not improve from 0.82440
Epoch 26/50
390/390 [==============================] - 33s 86ms/step - loss: 0.4302 - acc: 0.8519 - val_loss: 0.5550 - val_acc: 0.8118

Epoch 00026: val_acc did not improve from 0.82440
Epoch 27/50
390/390 [==============================] - 33s 86ms/step - loss: 0.4235 - acc: 0.8551 - val_loss: 0.5091 - val_acc: 0.8266

Epoch 00027: val_acc improved from 0.82440 to 0.82660, saving model to /opt/saved_weight_file.hdf5
Epoch 28/50
390/390 [==============================] - 33s 85ms/step - loss: 0.4170 - acc: 0.8568 - val_loss: 0.5281 - val_acc: 0.8260

Epoch 00028: val_acc did not improve from 0.82660
Epoch 29/50
390/390 [==============================] - 33s 86ms/step - loss: 0.4097 - acc: 0.8593 - val_loss: 0.5954 - val_acc: 0.7968

Epoch 00029: val_acc did not improve from 0.82660
Epoch 30/50
390/390 [==============================] - 33s 86ms/step - loss: 0.4072 - acc: 0.8611 - val_loss: 0.5767 - val_acc: 0.8056

Epoch 00030: val_acc did not improve from 0.82660
Epoch 31/50
390/390 [==============================] - 33s 86ms/step - loss: 0.4000 - acc: 0.8624 - val_loss: 0.6637 - val_acc: 0.7818

Epoch 00031: val_acc did not improve from 0.82660
Epoch 32/50
390/390 [==============================] - 33s 86ms/step - loss: 0.3967 - acc: 0.8621 - val_loss: 0.5157 - val_acc: 0.8264

Epoch 00032: val_acc did not improve from 0.82660
Epoch 33/50
390/390 [==============================] - 34s 86ms/step - loss: 0.3545 - acc: 0.8776 - val_loss: 0.4436 - val_acc: 0.8492

Epoch 00033: val_acc improved from 0.82660 to 0.84920, saving model to /opt/saved_weight_file.hdf5
Epoch 34/50
390/390 [==============================] - 33s 86ms/step - loss: 0.3429 - acc: 0.8822 - val_loss: 0.4361 - val_acc: 0.8525

Epoch 00034: val_acc improved from 0.84920 to 0.85250, saving model to /opt/saved_weight_file.hdf5
Epoch 35/50
390/390 [==============================] - 33s 86ms/step - loss: 0.3404 - acc: 0.8844 - val_loss: 0.4309 - val_acc: 0.8539

Epoch 00035: val_acc improved from 0.85250 to 0.85390, saving model to /opt/saved_weight_file.hdf5
Epoch 36/50
390/390 [==============================] - 33s 86ms/step - loss: 0.3327 - acc: 0.8853 - val_loss: 0.4346 - val_acc: 0.8505

Epoch 00036: val_acc did not improve from 0.85390
Epoch 37/50
390/390 [==============================] - 34s 86ms/step - loss: 0.3339 - acc: 0.8864 - val_loss: 0.4439 - val_acc: 0.8505

Epoch 00037: val_acc did not improve from 0.85390
Epoch 38/50
390/390 [==============================] - 33s 86ms/step - loss: 0.3270 - acc: 0.8884 - val_loss: 0.4193 - val_acc: 0.8600

Epoch 00038: val_acc improved from 0.85390 to 0.86000, saving model to /opt/saved_weight_file.hdf5
Epoch 39/50
390/390 [==============================] - 33s 86ms/step - loss: 0.3288 - acc: 0.8893 - val_loss: 0.4376 - val_acc: 0.8556

Epoch 00039: val_acc did not improve from 0.86000
Epoch 40/50
390/390 [==============================] - 33s 86ms/step - loss: 0.3286 - acc: 0.8872 - val_loss: 0.4500 - val_acc: 0.8497

Epoch 00040: val_acc did not improve from 0.86000
Epoch 41/50
390/390 [==============================] - 33s 86ms/step - loss: 0.3210 - acc: 0.8907 - val_loss: 0.4194 - val_acc: 0.8597

Epoch 00041: val_acc did not improve from 0.86000
Epoch 42/50
390/390 [==============================] - 34s 86ms/step - loss: 0.3216 - acc: 0.8900 - val_loss: 0.4255 - val_acc: 0.8594

Epoch 00042: val_acc did not improve from 0.86000
Epoch 43/50
390/390 [==============================] - 34s 86ms/step - loss: 0.3181 - acc: 0.8897 - val_loss: 0.4180 - val_acc: 0.8610

Epoch 00043: val_acc improved from 0.86000 to 0.86100, saving model to /opt/saved_weight_file.hdf5
Epoch 44/50
390/390 [==============================] - 34s 86ms/step - loss: 0.3155 - acc: 0.8921 - val_loss: 0.4458 - val_acc: 0.8512

Epoch 00044: val_acc did not improve from 0.86100
Epoch 45/50
390/390 [==============================] - 33s 86ms/step - loss: 0.3161 - acc: 0.8907 - val_loss: 0.4163 - val_acc: 0.8596

Epoch 00045: val_acc did not improve from 0.86100
Epoch 46/50
390/390 [==============================] - 33s 86ms/step - loss: 0.3098 - acc: 0.8964 - val_loss: 0.4276 - val_acc: 0.8541

Epoch 00046: val_acc did not improve from 0.86100
Epoch 47/50
390/390 [==============================] - 34s 86ms/step - loss: 0.3079 - acc: 0.8949 - val_loss: 0.4664 - val_acc: 0.8456

Epoch 00047: val_acc did not improve from 0.86100
Epoch 48/50
390/390 [==============================] - 33s 86ms/step - loss: 0.3094 - acc: 0.8939 - val_loss: 0.4418 - val_acc: 0.8518

Epoch 00048: val_acc did not improve from 0.86100
Epoch 49/50
390/390 [==============================] - 34s 86ms/step - loss: 0.3117 - acc: 0.8934 - val_loss: 0.4258 - val_acc: 0.8587

Epoch 00049: val_acc did not improve from 0.86100
Epoch 50/50
390/390 [==============================] - 33s 86ms/step - loss: 0.3077 - acc: 0.8939 - val_loss: 0.4280 - val_acc: 0.8578

Epoch 00050: val_acc did not improve from 0.86100
Model took 1681.38 seconds to train

