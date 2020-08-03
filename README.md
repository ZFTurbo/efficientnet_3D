# EfficientNet 3D Keras (and TF.Keras)

The repository contains 3D variants of EfficientNet models for classification. 

This repository is based on great [efficientnet](https://github.com/qubvel/efficientnet) repo by [@qubvel](https://github.com/qubvel/)

### Requirements

* keras >= 2.2.0 and tensorflow >= 1.13
* keras_applications >= 1.0.7

### Installation

`pip install efficientnet-3D` 

### Examples 

##### Loading model:

```python
# models can be build with Keras or Tensorflow frameworks
# use keras and tfkeras modules respectively
# efficientnet.keras / efficientnet.tfkeras
import efficientnet_3D.keras as efn 
# import efficientnet_3D.tfkeras as efn

model = efn.EfficientNetB0(input_shape=(64, 64, 64, 3), weights='imagenet')
```

### Related repositories

 * [https://github.com/qubvel/classification_models](https://github.com/qubvel/classification_models) - original classification 2D repo
 * [https://github.com/qubvel/segmentation_models](https://github.com/qubvel/segmentation_models) - original segmentation 2D repo
 * [classification_models_3D](https://github.com/ZFTurbo/classification_models_3D) - models for classification in 3D
 * [volumentations](https://github.com/ZFTurbo/volumentations) - 3D augmentations
 
### Unresolved problems

* There is no DepthwiseConv3D layer in keras, so repo used custom layer from [this repo](https://github.com/alexandrosstergiou/keras-DepthwiseConv3D) by [@alexandrosstergiou]( https://github.com/alexandrosstergiou/keras-DepthwiseConv3D).
