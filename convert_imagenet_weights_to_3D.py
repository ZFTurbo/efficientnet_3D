# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


try:
    # keras
    from keras import backend as K
    print('Use keras...')
except:
    # tf keras
    from tensorflow.keras import backend as K
    print('Use TF keras...')
import glob
import hashlib


OUTPUT_PATH_CONVERTER = 'converter/'
if not os.path.isdir(OUTPUT_PATH_CONVERTER):
    os.mkdir(OUTPUT_PATH_CONVERTER)


def get_model_memory_usage(batch_size, model):
    import numpy as np

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def convert_weights(m2, m3, out_path, target_channel):
    print('Start: {}'.format(m2.name))
    for i in range(len(m2.layers)):
        layer_2D = m2.layers[i]
        layer_3D = m3.layers[i]
        print('Extract for [{}]: {} {}'.format(i, layer_2D.__class__.__name__, layer_2D.name))
        print('Set for [{}]: {} {}'.format(i, layer_3D.__class__.__name__, layer_3D.name))

        if layer_2D.name != layer_3D.name:
            print('Warning: different names!')

        weights_2D = layer_2D.get_weights()
        weights_3D = layer_3D.get_weights()
        if layer_2D.__class__.__name__ == 'Conv2D' or \
                layer_2D.__class__.__name__ == 'DepthwiseConv2D':
            print(type(weights_2D), len(weights_2D), weights_2D[0].shape, weights_3D[0].shape)
            weights_3D[0][...] = 0

            if target_channel == 2:
                for j in range(weights_3D[0].shape[2]):
                    weights_3D[0][:, :, j, :, :] = weights_2D[0] / weights_3D[0].shape[2]
            if target_channel == 1:
                for j in range(weights_3D[0].shape[1]):
                    weights_3D[0][:, j, :, :, :] = weights_2D[0] / weights_3D[0].shape[1]
            else:
                for j in range(weights_3D[0].shape[0]):
                    weights_3D[0][j, :, :, :, :] = weights_2D[0] / weights_3D[0].shape[0]
            m3.layers[i].set_weights(weights_3D)
        else:
            m3.layers[i].set_weights(weights_2D)
    m3.save(out_path)


def get_effnet_model(type_dim, type_eff, include_top):
    try:
        from efficientnet_3D.keras import EfficientNetB0 as EfficientNetB0_3D
        from efficientnet_3D.keras import EfficientNetB1 as EfficientNetB1_3D
        from efficientnet_3D.keras import EfficientNetB2 as EfficientNetB2_3D
        from efficientnet_3D.keras import EfficientNetB3 as EfficientNetB3_3D
        from efficientnet_3D.keras import EfficientNetB4 as EfficientNetB4_3D
        from efficientnet_3D.keras import EfficientNetB5 as EfficientNetB5_3D
        from efficientnet_3D.keras import EfficientNetB6 as EfficientNetB6_3D
        from efficientnet_3D.keras import EfficientNetB7 as EfficientNetB7_3D
        from efficientnet.keras import EfficientNetB0 as EfficientNetB0_2D
        from efficientnet.keras import EfficientNetB1 as EfficientNetB1_2D
        from efficientnet.keras import EfficientNetB2 as EfficientNetB2_2D
        from efficientnet.keras import EfficientNetB3 as EfficientNetB3_2D
        from efficientnet.keras import EfficientNetB4 as EfficientNetB4_2D
        from efficientnet.keras import EfficientNetB5 as EfficientNetB5_2D
        from efficientnet.keras import EfficientNetB6 as EfficientNetB6_2D
        from efficientnet.keras import EfficientNetB7 as EfficientNetB7_2D
    except:
        from efficientnet_3D.tfkeras import EfficientNetB0 as EfficientNetB0_3D
        from efficientnet_3D.tfkeras import EfficientNetB1 as EfficientNetB1_3D
        from efficientnet_3D.tfkeras import EfficientNetB2 as EfficientNetB2_3D
        from efficientnet_3D.tfkeras import EfficientNetB3 as EfficientNetB3_3D
        from efficientnet_3D.tfkeras import EfficientNetB4 as EfficientNetB4_3D
        from efficientnet_3D.tfkeras import EfficientNetB5 as EfficientNetB5_3D
        from efficientnet_3D.tfkeras import EfficientNetB6 as EfficientNetB6_3D
        from efficientnet_3D.tfkeras import EfficientNetB7 as EfficientNetB7_3D
        from efficientnet.tfkeras import EfficientNetB0 as EfficientNetB0_2D
        from efficientnet.tfkeras import EfficientNetB1 as EfficientNetB1_2D
        from efficientnet.tfkeras import EfficientNetB2 as EfficientNetB2_2D
        from efficientnet.tfkeras import EfficientNetB3 as EfficientNetB3_2D
        from efficientnet.tfkeras import EfficientNetB4 as EfficientNetB4_2D
        from efficientnet.tfkeras import EfficientNetB5 as EfficientNetB5_2D
        from efficientnet.tfkeras import EfficientNetB6 as EfficientNetB6_2D
        from efficientnet.tfkeras import EfficientNetB7 as EfficientNetB7_2D

    shape_size_3D = (32, 32, 224, 3)
    shape_size_2D = (224, 224, 3)

    if type_dim == '2D':
        if type_eff == 'efficientnet-b0':
            model = EfficientNetB0_2D(include_top=include_top,
                                    weights='imagenet',
                                    input_shape=shape_size_2D,
                                    pooling='avg', )
        elif type_eff == 'efficientnet-b1':
            model = EfficientNetB1_2D(include_top=include_top,
                                    weights='imagenet',
                                    input_shape=shape_size_2D,
                                    pooling='avg', )
        elif type_eff == 'efficientnet-b2':
            model = EfficientNetB2_2D(include_top=include_top,
                                    weights='imagenet',
                                    input_shape=shape_size_2D,
                                    pooling='avg', )
        elif type_eff == 'efficientnet-b3':
            model = EfficientNetB3_2D(include_top=include_top,
                                    weights='imagenet',
                                    input_shape=shape_size_2D,
                                    pooling='avg', )
        elif type_eff == 'efficientnet-b4':
            model = EfficientNetB4_2D(include_top=include_top,
                                    weights='imagenet',
                                    input_shape=shape_size_2D,
                                    pooling='avg', )
        elif type_eff == 'efficientnet-b5':
            model = EfficientNetB5_2D(include_top=include_top,
                                    weights='imagenet',
                                    input_shape=shape_size_2D,
                                    pooling='avg', )
        elif type_eff == 'efficientnet-b6':
            model = EfficientNetB6_2D(include_top=include_top,
                                    weights='imagenet',
                                    input_shape=shape_size_2D,
                                    pooling='avg', )
        elif type_eff == 'efficientnet-b7':
            model = EfficientNetB7_2D(include_top=include_top,
                                    weights='imagenet',
                                    input_shape=shape_size_2D,
                                    pooling='avg', )
    else:
        if type_eff == 'efficientnet-b0':
            model = EfficientNetB0_3D(include_top=False,
                                        weights=None,
                                        input_shape=shape_size_3D,
                                        pooling='avg', )
        elif type_eff == 'efficientnet-b1':
            model = EfficientNetB1_3D(include_top=False,
                                        weights=None,
                                        input_shape=shape_size_3D,
                                        pooling='avg', )
        elif type_eff == 'efficientnet-b2':
            model = EfficientNetB2_3D(include_top=False,
                                        weights=None,
                                        input_shape=shape_size_3D,
                                        pooling='avg', )
        elif type_eff == 'efficientnet-b3':
            model = EfficientNetB3_3D(include_top=False,
                                        weights=None,
                                        input_shape=shape_size_3D,
                                        pooling='avg', )
        elif type_eff == 'efficientnet-b4':
            model = EfficientNetB4_3D(include_top=False,
                                        weights=None,
                                        input_shape=shape_size_3D,
                                        pooling='avg', )
        elif type_eff == 'efficientnet-b5':
            model = EfficientNetB5_3D(include_top=False,
                                        weights=None,
                                        input_shape=shape_size_3D,
                                        pooling='avg', )
        elif type_eff == 'efficientnet-b6':
            model = EfficientNetB6_3D(include_top=False,
                                        weights=None,
                                        input_shape=shape_size_3D,
                                        pooling='avg', )
        elif type_eff == 'efficientnet-b7':
            model = EfficientNetB7_3D(include_top=False,
                                        weights=None,
                                        input_shape=shape_size_3D,
                                        pooling='avg', )

    return model


def convert_model_effnet():
    target_channel = 0
    include_top = False
    list_to_check = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                     'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
    for t in list_to_check:
        model3D = get_effnet_model('3D', t, include_top)
        mem = get_model_memory_usage(1, model3D)
        print('Model 3D: {} Mem single: {:.2f}'.format(t, mem))
        model2D = get_effnet_model('2D', t, include_top)
        mem = get_model_memory_usage(1, model2D)
        print('Model 3D: {} Mem single: {:.2f}'.format(t, mem))

        out_path = OUTPUT_PATH_CONVERTER + '{}_inp_channel_{}_tch_{}_top_{}.h5'.format(t, 3, target_channel, include_top)
        convert_weights(model2D, model3D, out_path, target_channel=target_channel)


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def gen_text_with_links():
    list_to_check = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                     'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
    print('{')
    for model_name in list_to_check:
        files = glob.glob('./converter/{}_*.h5'.format(model_name))
        for f in files:
            m5 = md5(f)
            print('    \'{}\': (\'{}\', ),'.format(model_name, m5))
    print('}')


if __name__ == '__main__':
    # convert_model_effnet()
    gen_text_with_links()