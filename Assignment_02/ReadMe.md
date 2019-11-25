### 1. Log for 20 epochs of training

Epoch 1/20
60000/60000 [==============================] - 27s 451us/step - loss: 0.3196 - acc: 0.9511 - val_loss: 0.0872 - val_acc: 0.9873

Epoch 00001: val_acc improved from -inf to 0.98730, saving model to /opt/saved_weight_file.hdf5
Epoch 2/20
60000/60000 [==============================] - 22s 371us/step - loss: 0.1148 - acc: 0.9800 - val_loss: 0.0436 - val_acc: 0.9926

Epoch 00002: val_acc improved from 0.98730 to 0.99260, saving model to /opt/saved_weight_file.hdf5
Epoch 3/20
60000/60000 [==============================] - 22s 369us/step - loss: 0.0735 - acc: 0.9853 - val_loss: 0.0326 - val_acc: 0.9931

Epoch 00003: val_acc improved from 0.99260 to 0.99310, saving model to /opt/saved_weight_file.hdf5
Epoch 4/20
60000/60000 [==============================] - 22s 373us/step - loss: 0.0516 - acc: 0.9892 - val_loss: 0.0264 - val_acc: 0.9938

Epoch 00004: val_acc improved from 0.99310 to 0.99380, saving model to /opt/saved_weight_file.hdf5
Epoch 5/20
60000/60000 [==============================] - 23s 377us/step - loss: 0.0378 - acc: 0.9918 - val_loss: 0.0241 - val_acc: 0.9935

Epoch 00005: val_acc did not improve from 0.99380
Epoch 6/20
60000/60000 [==============================] - 23s 376us/step - loss: 0.0324 - acc: 0.9924 - val_loss: 0.0221 - val_acc: 0.9927

Epoch 00006: val_acc did not improve from 0.99380
Epoch 7/20
60000/60000 [==============================] - 22s 369us/step - loss: 0.0261 - acc: 0.9938 - val_loss: 0.0210 - val_acc: 0.9934

Epoch 00007: val_acc did not improve from 0.99380
Epoch 8/20
60000/60000 [==============================] - 22s 370us/step - loss: 0.0217 - acc: 0.9949 - val_loss: 0.0284 - val_acc: 0.9916

Epoch 00008: val_acc did not improve from 0.99380
Epoch 9/20
60000/60000 [==============================] - 22s 369us/step - loss: 0.0196 - acc: 0.9953 - val_loss: 0.0205 - val_acc: 0.9941

Epoch 00009: val_acc improved from 0.99380 to 0.99410, saving model to /opt/saved_weight_file.hdf5
Epoch 10/20
60000/60000 [==============================] - 22s 370us/step - loss: 0.0180 - acc: 0.9957 - val_loss: 0.0197 - val_acc: 0.9943

Epoch 00010: val_acc improved from 0.99410 to 0.99430, saving model to /opt/saved_weight_file.hdf5
Epoch 11/20
60000/60000 [==============================] - 22s 370us/step - loss: 0.0158 - acc: 0.9960 - val_loss: 0.0171 - val_acc: 0.9949

Epoch 00011: val_acc improved from 0.99430 to 0.99490, saving model to /opt/saved_weight_file.hdf5
Epoch 12/20
60000/60000 [==============================] - 22s 370us/step - loss: 0.0137 - acc: 0.9968 - val_loss: 0.0241 - val_acc: 0.9930

Epoch 00012: val_acc did not improve from 0.99490
Epoch 13/20
60000/60000 [==============================] - 22s 368us/step - loss: 0.0125 - acc: 0.9971 - val_loss: 0.0209 - val_acc: 0.9932

Epoch 00013: val_acc did not improve from 0.99490
Epoch 14/20
60000/60000 [==============================] - 22s 373us/step - loss: 0.0110 - acc: 0.9973 - val_loss: 0.0245 - val_acc: 0.9934

Epoch 00014: val_acc did not improve from 0.99490
Epoch 15/20
60000/60000 [==============================] - 22s 371us/step - loss: 0.0103 - acc: 0.9976 - val_loss: 0.0218 - val_acc: 0.9938

Epoch 00015: val_acc did not improve from 0.99490
Epoch 16/20
60000/60000 [==============================] - 22s 369us/step - loss: 0.0091 - acc: 0.9976 - val_loss: 0.0185 - val_acc: 0.9948

Epoch 00016: val_acc did not improve from 0.99490
Epoch 17/20
60000/60000 [==============================] - 22s 368us/step - loss: 0.0044 - acc: 0.9991 - val_loss: 0.0179 - val_acc: 0.9949

Epoch 00017: val_acc did not improve from 0.99490
Epoch 18/20
60000/60000 [==============================] - 22s 370us/step - loss: 0.0029 - acc: 0.9995 - val_loss: 0.0163 - val_acc: 0.9957

Epoch 00018: val_acc improved from 0.99490 to 0.99570, saving model to /opt/saved_weight_file.hdf5
Epoch 19/20
60000/60000 [==============================] - 23s 376us/step - loss: 0.0029 - acc: 0.9994 - val_loss: 0.0184 - val_acc: 0.9950

Epoch 00019: val_acc did not improve from 0.99570
Epoch 20/20
60000/60000 [==============================] - 22s 371us/step - loss: 0.0025 - acc: 0.9995 - val_loss: 0.0152 - val_acc: 0.9957

Epoch 00020: val_acc did not improve from 0.99570

### 2. Log of evaluate

[0.01630240877982578, 0.9957]

### 3. Strategy

* Number of parameters are 80k approx.  
* Before transition block, there are three convolution layers of 3x3.  
* After transition block, there are three convolution layers of 3x3.  
* Flatten is replaced by GlobalAveragePool2D.  
* BatchNormalization is placed after each regular convolution layer.  
* Dropouts are placed in such a way that it is decreasing.  
  i.e., beginning few layers have dropout of 0.1 and layer layers have 0.01. 
  
  ### File name
EIP4_Assignment2_10_good.ipynb
