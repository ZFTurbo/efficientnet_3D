# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


def tst_keras():
    # for keras
    try:
        import efficientnet_3D.keras as efn
    except:
        import efficientnet_3D.tfkeras as efn

    weights = 'imagenet'
    model = efn.EfficientNetB0(input_shape=(64, 64, 64, 3), weights=weights, pooling='avg')
    print(model.summary())
    model = efn.EfficientNetB1(input_shape=(64, 64, 64, 3), weights=weights)
    print(model.summary())
    model = efn.EfficientNetB2(input_shape=(64, 64, 64, 3), weights=weights)
    print(model.summary())
    model = efn.EfficientNetB3(input_shape=(64, 64, 64, 3), weights=weights)
    print(model.summary())
    model = efn.EfficientNetB4(input_shape=(64, 64, 64, 3), weights=weights)
    print(model.summary())
    model = efn.EfficientNetB5(input_shape=(64, 64, 64, 3), weights=weights)
    print(model.summary())
    model = efn.EfficientNetB6(input_shape=(64, 64, 64, 3), weights=weights)
    print(model.summary())
    model = efn.EfficientNetB7(input_shape=(64, 64, 64, 3), weights=weights)
    print(model.summary())


if __name__ == '__main__':
    tst_keras()
