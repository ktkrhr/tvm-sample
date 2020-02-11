#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np

from PIL import Image

import keras
from keras.applications.mobilenet import preprocess_input

from tvm import relay
#from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime

WEIGHT_FILE = 'mobilenet_1_0_224_tf.h5'
LABELS = 'class_labels.txt'

NUM_LOOP = 10
TOP_N = 5


def run(input_data, verbose, cpu):
    with open(LABELS) as f:
        labels = f.read().split('\n')

    batch_size = 1
    num_class = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size, ) + image_shape
    out_shape = (batch_size, num_class)

    if not os.path.exists(WEIGHT_FILE):
        print(f'{WEIGHT_FILE} does not exist.')
        print('Download keras mobilenet v1 model from https://github.com/fchollet/deep-learning-models/releases)')
        sys.exit(0)

    # keras model load
    model = keras.applications.mobilenet.MobileNet(
        include_top=True, weights=None,
        input_shape=(224, 224, 3), classes=1000)
    model.load_weights(WEIGHT_FILE)

    # input data loading
    with Image.open(input_data) as img:
        img = img.resize((224, 224), resample=Image.BILINEAR)
        data = np.array(img)[np.newaxis, :].astype('float32')
        data = preprocess_input(data)
        data = data.transpose((0, 3, 1, 2))

    # relay model loading with Keras frontend
    shape_dict = {'input_1': data.shape}
    mod, params = relay.frontend.from_keras(model, shape_dict)
    '''
    # for relay.testing without pretrained weights
    mod, params = relay.testing.mobilenet.get_workload(
        batch_size=batch_size, num_classes=num_class,
        image_shape=image_shape, dtype='float32', layout='NCHW')
    '''

    if verbose:
        print(mod.astext(show_meta_data=False))

    # optimization and compilation
    opt_level = 3
    if not cpu:
        target = tvm.target.cuda(model='1080ti')
    else:
        target = tvm.target.create('llvm')
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build_module.build(
            mod, target, params=params)

    # run inference
    if not cpu:
        ctx = tvm.gpu()
    else:
        ctx = tvm.cpu()

    module = graph_runtime.create(graph, lib, ctx)

    module.set_input('input_1', data)
    module.set_input(**params)

    module.run()
    output = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()

    output = output.flatten()
    pred_idx = np.argsort(output)[::-1]
    pred_prob = np.sort(output)[::-1]

    print('\nClassification Result:')
    for i in range(TOP_N):
        print(f'{i+1} {labels[pred_idx[i]]} {pred_prob[i]:.6f}')
    
    # benchmark
    print('\nEvaluate inference time cost...')
    ftimer = module.module.time_evaluator('run', ctx, number=1, repeat=10)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    for i in range(len(prof_res)):
        print(f'Inference time {i}: {prof_res[i]:.6f}')
    print('Mean inference time (std dev): {0:.6f} ms ({1:.6f} ms)'.format(np.mean(prof_res), np.std(prof_res)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input image')
    parser.add_argument('--cpu', action='store_true',
                        help='cpu inference')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='enable verbose logging')
                        
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    run(args.input, args.verbose, args.cpu)
