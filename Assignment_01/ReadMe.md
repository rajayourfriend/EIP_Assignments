## Definitions


### Convolution
        The operation of multiplication and addition combined together is known as convolution. Normally the input image or channel is multiplied and added with kernel. 

### Filters/Kernels
        The filter or kernel is known as feature extractor. Normally its size is 3x3. When input image / channel is convolved with kernel, we get feature map, 
    that shall convey something that is understandable to neural network.

### Epochs
        The epoch is nothing but the exhaust of all the input samples being fed to the neural network. When all the input samples are fed atleast once to the network 
    during training, we term it as one epoch. 

### 1x1 Convolution
        The convolution with 1x1 on a channel consolidates the information that is extracted from data. It may be used to reduce the size of channel or increase size of channel.

### 3x3 Convolution
        The convolution with 3x3 on a channel extracts the feature available(information) in data. It is also known as filter/kernel/feature extractor.

### Feature Maps
        The result of convolution of a kernel on input image / channel is called feature maps.

### Activation Function
        Since convolution brings in linearity, we need another guy called activation function that brings non-linearity to the neural network. 
    Hence the combination of convolution and activation function forms the heart of neural network. Rectified linear unit(relu), tanh, sigmoid are the popularly used activation functions.

### Receptive Field
        The extent to which the image has been visualized by the network is known as receptive field. As the layer is incremented (go deeper), the receptive field keeps increasing.
    A kernel of bigger dimension will have higher receptive field in a layer. A kernel of smaller dimension will have lower receptive field in a layer.
