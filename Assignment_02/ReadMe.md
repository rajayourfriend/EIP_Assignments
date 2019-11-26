### 1. Log for 20 epochs of training

Epoch 1/20
60000/60000 [==============================] - 39s 643us/step - loss: 0.4536 - acc: 0.9077 - val_loss: 0.1073 - val_acc: 0.9826

Epoch 00001: val_acc improved from -inf to 0.98260, saving model to /opt/saved_weight_file.hdf5
Epoch 2/20
60000/60000 [==============================] - 34s 567us/step - loss: 0.1580 - acc: 0.9674 - val_loss: 0.0621 - val_acc: 0.9871

Epoch 00002: val_acc improved from 0.98260 to 0.98710, saving model to /opt/saved_weight_file.hdf5
Epoch 3/20
60000/60000 [==============================] - 34s 559us/step - loss: 0.1098 - acc: 0.9748 - val_loss: 0.0419 - val_acc: 0.9904

Epoch 00003: val_acc improved from 0.98710 to 0.99040, saving model to /opt/saved_weight_file.hdf5
Epoch 4/20
60000/60000 [==============================] - 34s 560us/step - loss: 0.0837 - acc: 0.9804 - val_loss: 0.0341 - val_acc: 0.9919

Epoch 00004: val_acc improved from 0.99040 to 0.99190, saving model to /opt/saved_weight_file.hdf5
Epoch 5/20
60000/60000 [==============================] - 34s 563us/step - loss: 0.0697 - acc: 0.9828 - val_loss: 0.0326 - val_acc: 0.9908

Epoch 00005: val_acc did not improve from 0.99190
Epoch 6/20
60000/60000 [==============================] - 34s 564us/step - loss: 0.0592 - acc: 0.9850 - val_loss: 0.0284 - val_acc: 0.9922

Epoch 00006: val_acc improved from 0.99190 to 0.99220, saving model to /opt/saved_weight_file.hdf5
Epoch 7/20
60000/60000 [==============================] - 34s 563us/step - loss: 0.0536 - acc: 0.9864 - val_loss: 0.0268 - val_acc: 0.9934

Epoch 00007: val_acc improved from 0.99220 to 0.99340, saving model to /opt/saved_weight_file.hdf5
Epoch 8/20
60000/60000 [==============================] - 34s 564us/step - loss: 0.0477 - acc: 0.9870 - val_loss: 0.0251 - val_acc: 0.9939

Epoch 00008: val_acc improved from 0.99340 to 0.99390, saving model to /opt/saved_weight_file.hdf5
Epoch 9/20
60000/60000 [==============================] - 34s 563us/step - loss: 0.0450 - acc: 0.9877 - val_loss: 0.0243 - val_acc: 0.9933

Epoch 00009: val_acc did not improve from 0.99390
Epoch 10/20
60000/60000 [==============================] - 34s 563us/step - loss: 0.0418 - acc: 0.9887 - val_loss: 0.0266 - val_acc: 0.9927

Epoch 00010: val_acc did not improve from 0.99390
Epoch 11/20
60000/60000 [==============================] - 34s 567us/step - loss: 0.0400 - acc: 0.9891 - val_loss: 0.0204 - val_acc: 0.9935

Epoch 00011: val_acc did not improve from 0.99390
Epoch 12/20
60000/60000 [==============================] - 34s 563us/step - loss: 0.0342 - acc: 0.9906 - val_loss: 0.0243 - val_acc: 0.9932

Epoch 00012: val_acc did not improve from 0.99390
Epoch 13/20
60000/60000 [==============================] - 34s 561us/step - loss: 0.0343 - acc: 0.9908 - val_loss: 0.0217 - val_acc: 0.9934

Epoch 00013: val_acc did not improve from 0.99390
Epoch 14/20
60000/60000 [==============================] - 34s 563us/step - loss: 0.0336 - acc: 0.9910 - val_loss: 0.0254 - val_acc: 0.9925

Epoch 00014: val_acc did not improve from 0.99390
Epoch 15/20
60000/60000 [==============================] - 34s 562us/step - loss: 0.0332 - acc: 0.9907 - val_loss: 0.0240 - val_acc: 0.9927

Epoch 00015: val_acc did not improve from 0.99390
Epoch 16/20
60000/60000 [==============================] - 34s 563us/step - loss: 0.0318 - acc: 0.9912 - val_loss: 0.0216 - val_acc: 0.9935

Epoch 00016: val_acc did not improve from 0.99390
Epoch 17/20
60000/60000 [==============================] - 34s 562us/step - loss: 0.0228 - acc: 0.9934 - val_loss: 0.0189 - val_acc: 0.9942

Epoch 00017: val_acc improved from 0.99390 to 0.99420, saving model to /opt/saved_weight_file.hdf5
Epoch 18/20
60000/60000 [==============================] - 34s 564us/step - loss: 0.0207 - acc: 0.9945 - val_loss: 0.0173 - val_acc: 0.9949

Epoch 00018: val_acc improved from 0.99420 to 0.99490, saving model to /opt/saved_weight_file.hdf5
Epoch 19/20
60000/60000 [==============================] - 34s 562us/step - loss: 0.0215 - acc: 0.9943 - val_loss: 0.0189 - val_acc: 0.9949

Epoch 00019: val_acc did not improve from 0.99490
Epoch 20/20
60000/60000 [==============================] - 34s 560us/step - loss: 0.0201 - acc: 0.9946 - val_loss: 0.0179 - val_acc: 0.9944

Epoch 00020: val_acc did not improve from 0.99490  


### 2. Log of evaluate

[0.01727237341988366, 0.9949]

### 3. Strategy  

* Number of parameters are 12k approx.  
* val_acc = 99.49%
* Before transition block, there are three convolution layers of 3x3.  
* After transition block, there are three convolution layers of 3x3.  
* Flatten is replaced by GlobalAveragePool2D.  
* ReduceLROnPlateau is used for updating Learning Rate when it is found in plateau
* BatchNormalization is placed after each regular convolution layer.  
* Dropouts are placed in such a way that it is decreasing.  
  i.e., beginning many layers have dropout of 0.1 and fifth conv layer has 0.01. After that no dropout.

