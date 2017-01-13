### Overall Trend of shape:
* Image Shape decrease (227 >> 55 >> 27 >> 13 >> 6)
* Depth increase (3 >> 96 >> 256 >> 384 >> 256)
* Neuron Count:
    * CONV1: 55 x 55 x 96
    * CONV2: 27 x 27 x 256
    * CONV3: 13 x 13 x 384
    * CONV4: 13 x 13 x 384
    * CONV5: 13 x 13 x 256
    * FC1: 4096
    * FC2: 4096
    * FC3: 1000

* Parameter size:
    * CONV1: (11 x 11 x 3 + 1) x (55 x 55 x 96) = 105,705,600
    * CONV2: (5 x 5 x 96 + 1) x (27 x 27 x 256) = 448,084,224
    * CONV3: (3 x 3 x 256 + 1) x (13 x 13 x 384)
    * CONV4: (3 x 3 x 384 + 1) x (13 x 13 x 384) = 224345472
    * CONV5: (3 x 3 x 384 + 1) x (13 x 13 x 256) = 149,520,384
    * FC1: (6 x 6 x 256 + 1) x 4096 = 37,752,832
    * FC2: (4096 + 1) x 4096 = 16,781,312
    * FC3: (4096 + 1) x 1000 = 4,097,000

### Layer Parameters
* Layer 0: Input image
    * Size: 227 x 227 x 3
    * Kernel: None
    * Weight: None
    * Bias: None
    * Note that in the paper referenced above, the network diagram has 224x224x3 printed which appears to be a typo.
* Layer 1 (CONV1): Convolution with 96 filters, size 11×11, stride 4, padding 0
    * Size: 55 x 55 x 96
    * Kernel: 11x11
    * Weight: 11x11x3 x 96
    * Bias: 1 x 96
    * (227-11)/4 + 1 = 55 is the size of the outcome
    * 96 depth because 1 set denotes 1 filter and there are 96 filters
* Layer 2: Max-Pooling with 3×3 filter, stride 2
    * Size: 27 x 27 x 96
    * (55 – 3)/2 + 1 = 27 is size of outcome
    * depth is same as before, i.e. 96 because pooling is done independently on each layer
* Layer 3 (CONV2): Convolution with 256 filters, size 5×5, stride 1, padding 2
    * Size: 27 x 27 x 256
    * Kernel: 5x5
    * Weight: 5x5x96 x 256
    * Bias: 1 x 256
    * Because of padding of (5-1)/2=2, the original size is restored
    * 256 depth because of 256 filters
* Layer 4: Max-Pooling with 3×3 filter, stride 2
    * Size: 13 x 13 x 256
    * (27 – 3)/2 + 1 = 13 is size of outcome
    * Depth is same as before, i.e. 256 because pooling is done independently on each layer
* Layer 5 (CONV3): Convolution with 384 filters, size 3×3, stride 1, padding 1
    * Size: 13 x 13 x 384
    * Kernel: 3x3
    * Weight: 3x3x256 x 384
    * Bias: 1 x 384
    * Because of padding of (3-1)/2=1, the original size is restored
    * 384 depth because of 384 filters
* Layer 6 (CONV4): Convolution with 384 filters, size 3×3, stride 1, padding 1
    * Size: 13 x 13 x 384
    * Kernel: 3x3
    * Weight: 3x3x384 x 384
    * Bias: 1 x 384
    * Because of padding of (3-1)/2=1, the original size is restored
    * 384 depth because of 384 filters
* Layer 7 (CONV5): Convolution with 256 filters, size 3×3, stride 1, padding 1
    * Size: 13 x 13 x 256
    * Kernel: 3x3
    * Weight: 3x3x384 x 256
    * Bias: 1 x 256
    * Because of padding of (3-1)/2=1, the original size is restored
    * 256 depth because of 256 filters
* Layer 8: Max-Pooling with 3×3 filter, stride 2
    * Size: 6 x 6 x 256
    * (13 – 3)/2 + 1 = 6 is size of outcome
    * Depth is same as before, i.e. 256 because pooling is done independently on each layer
* Layer 9 (FC1): Fully Connected with 4096 neuron
    * In this later, each of the 6x6x256=9216 pixels are fed into each of the 4096 neurons and weights determined by back-propagation.
* Layer 10 (FC2): Fully Connected with 4096 neuron
    * Similar to layer #9
* Layer 11 (FC3): Fully Connected with 1000 neurons
    * This is the last layer and has 1000 neurons because IMAGENET data has 1000 classes to be predicted.




