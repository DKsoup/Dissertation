#!/usr/bin/env python

import os
import sys
import argparse
import pprint
import warnings
import functools
import csv
import json
from datetime import datetime, timedelta
import time
import gc

import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator  #, load_img, img_to_array
# from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.applications import vgg16, vgg19, resnet_v2
from tensorflow.keras.metrics import (categorical_crossentropy, 
                                      categorical_accuracy, 
                                      top_k_categorical_accuracy)

print(os.getcwd())
pprint.pprint(sys.path)

# project_root_dir = os.path.dirname(os.path.realpath(__file__))
# print(project_root_dir)
# sys.exit()

# sys.path.append('/work/generalisation-humans-DNNs/code')
# sys.path.append('/work/generalisation-humans-DNNs/code/accuracy_evaluation/')
# sys.path.append('/work/code/keras_lr_finder/')
# from mappings import HumanCategories
from bionet.config import (#data_set, classes, n_classes, 
                           classes, n_classes,
                           luminance_weights,
                           colour, contrast_level,
                           upscale, image_size, image_shape, train_image_stats,
                           interpolation_names,
#                            data_dir, models_dir, logs_dir, results_dir,
                           max_queue_size, workers, use_multiprocessing,
                           report, extension,
                           generalisation_types, generalisation_sets, generalisation_columns)
from bionet import utils, plots
from bionet.preparation import (#as_perturbation_fn, as_greyscale_perturbation_fn, 
                                get_perturbations, stochastic_perturbations,
                                cifar_wrapper, get_noise_preprocessor, 
                                sanity_check,
                                uniform_noise, salt_and_pepper_noise, 
                                high_pass_filter, low_pass_filter,
                                adjust_contrast, scramble_phases,
                                rotate_image, adjust_brightness, 
                                invert_luminance)
from bionet.bases import BioResNet50, allcnn, allcnn_imagenet
from bionet.assess import test_noise_perturbations


# try:
#     from all_cnn.networks import allcnn, allcnn_imagenet
# except ImportError:
#     print("Please add an implementation of ALL-CNN to your path!")

# NOTE: Randomness and reproducibility
# https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
# It is difficult to obtain precisely reproducible results with Tensorflow (although improved in TF2).
# >> np.random.seed(seed)
# >> random.seed(seed)
# >> tf.set_random_seed(seed)
# >> os.environ['PYTHONHASHSEED'] = '0'
# However, when training on a GPU, the cuDNN stack introduces sources of "randomness" since the order of execution is not always guaranteed when running operations in parallel.
# Currently, there is no such attempt to make the results reproducible - only to ensure that each run is sufficiently different when testing with noise. 

# pprint.pprint(sys.path)
print('+' * 80)  # Simulation metadata
print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Starting simulation...')
print("\nTensorFlow:", tf.__version__)
print(f"Channel ordering: {tf.keras.backend.image_data_format()}")  # TensorFlow: Channels last order.
gpus = tf.config.experimental.list_physical_devices('GPU')
pprint.pprint(gpus)

# dtype = 'float16'  # Theoretically, not supported on Titan Xp
# tf.keras.backend.set_floatx(dtype)
# tf.keras.backend.set_epsilon(1e-4)  # Default 1e-7
# In practice, this works for testing models trained with float32 backend but slows testing by ~14%. 
# TODO: Try training on float16
# tf.keras.backend.set_floatx('float16')  # Set default dtype to 16 bit
print(f"Backend set to: {tf.keras.backend.floatx()}")  # Needed to stop OOM error with data_gen.flow() on TF>2.2

# warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data.", UserWarning)
warnings.filterwarnings("ignore", "tensorflow:Model failed to serialize as JSON.", Warning)


# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--convolution', type=str, default='Original',
                    help='Name of convolutional filter to use')
parser.add_argument('--base', type=str, default='VGG16',
                    help='Name of model to use')
parser.add_argument('--pretrain', action='store_true', # type=bool, default=False,
                    help='Flag to use pretrained ImageNet weights in the model')
parser.add_argument('--architecture', type=str, default='model.json',
                    help='Parameter file (JSON) to load')
# parser.add_argument('--upscale', action='store_true', #default=False, required=False,
#                     help='Flag to upscale the CIFAR10 images')
# parser.add_argument('--interpolate', action='store_true', default=False, required=False,
#                     help='Flag to interpolate the images when upscaling')
parser.add_argument('--interpolation', type=int, default=0,
                    help='Method to interpolate the images when upscaling. Default: 0 ("nearest" i.e. no interpolation)')
parser.add_argument('--optimizer', type=str, default='RMSprop',
                    choices=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
                    help='Name of optimizer to use: https://keras.io/optimizers/')
parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, required=False,  # '-r',  
                    help='Learning rate for training')
parser.add_argument('--decay', type=float, default=1e-6, required=False,
                    help='Optimizer decay for training')
parser.add_argument('--use_initializer', action='store_true', default=False, required=False,
                    help='Flag to use the weight initializer (then freeze weights) for the Gabor filters')
# parser.add_argument('--add_noise', action='store_true', default=False, required=False,
#                     help='Flag to add a Gaussian noise layer after the first convolutional layer')
parser.add_argument('--internal_noise', type=float, default=None, required=False,
                    help='Standard deviation for adding a Gaussian noise layer after the first convolutional layer')
parser.add_argument('--trial', type=int, default=1,  # default to 0 when unspecified?
                    help='Trial number for labeling different runs of the same model')
parser.add_argument('--label', type=str, default='',
                    help='For labeling different runs of the same model')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed to use')
parser.add_argument('-t', '--train', action='store_true', # type=bool, default=False,
                    help='Flag to train the model')
# This accompanying unfinished code was deleted in a single commit 25/6/20
# parser.add_argument('--train_with_noise', action='store_true',
#                     help='Flag to train the model with noise-like masks')
parser.add_argument('--recalculate_statistics', action='store_true', required=False,
                    help='Flag to recalculate normalisation statistics over the training set')
parser.add_argument('--epochs', type=int, default=20, required=False,
                    help='Number of epochs to train model')
parser.add_argument("--batch", type=int, default=64,
                    help="Size of mini-batches passed to the network")
# parser.add_argument('--data_set', type=str, default='CIFAR10', required=False,
#                     help="Predefined data set to use")
parser.add_argument('--image_path', type=str, default='',
                    help='Path to image files to load')
parser.add_argument('--train_image_path', type=str, default='',
                    help='Path to training image files to load')
# parser.add_argument('--test_image_path', type=str, default='',
#                     help='Path to testing image files to load')
parser.add_argument('--test_generalisation', action='store_true',
                    help='Flag to test the model on sets of untrained images')
parser.add_argument('--invert_test_images', type=bool, default=True, #action='store_true',
                    help='Flag to invert the luminance of the test images')
parser.add_argument('--test_perturbations', action='store_true',
                    help='Flag to test the model on perturbed images')
parser.add_argument('--data_augmentation', action='store_true', # type=bool, default=False,
                    help='Flag to train the model with data augmentation')
parser.add_argument('--extra_augmentation', action='store_true', # type=bool, default=False,
                    help='Flag to train the model with additional data augmentation')
parser.add_argument('-c', '--clean', action='store_true', default=False, required=False,
                    help='Flag to retrain model')
parser.add_argument('--skip_test', action='store_true',
                    help='Flag to skip testing the model')
parser.add_argument('-l', '--log', action='store_true', default=False, required=False,  # type=bool, 
                    help='Flag to log training data')
parser.add_argument('--save_images', action='store_true', default=False, required=False,
                    help='Flag to save preprocessed (perturbed) test images')
parser.add_argument('-p', '--save_predictions', action='store_true', default=False, required=False,  # type=bool, 
                    help='Flag to save category predictions')
parser.add_argument('--gpu', type=int, default=0, required=False,
                    help='GPU ID to run on')
parser.add_argument('--project_dir', type=str, default='',
                    help='Path to the root project directory')
parser.add_argument('-v', '--verbose', type=int, default=0, required=False,
                    help='Verbosity level')

args = vars(parser.parse_args())  # vars() returns a dict

if not args:
    parser.print_help()
    parser.exit(1)

# gpus = tf.config.experimental.list_physical_devices('GPU')
assert 0 <= args["gpu"] <= len(gpus)
tf.config.experimental.set_visible_devices(gpus[args["gpu"]], 'GPU')

convolution = args['convolution']
base = args['base']
# upscale = args['upscale']
# interpolate = args['interpolate']
interpolation = args['interpolation']
train = args['train']
clean = args['clean']
epochs = args['epochs']
batch = args['batch']  # 64  # 32
image_path = args['image_path']  # Deprecate?
train_image_path = args['train_image_path']
# test_image_path = args['test_image_path']
test_generalisation = args['test_generalisation']
invert_test_images = args['invert_test_images']
test_perturbations = args['test_perturbations']
data_augmentation = args['data_augmentation']
extra_augmentation = args['extra_augmentation']
recalculate_statistics = args['recalculate_statistics']
optimizer = args['optimizer']  # 'RMSprop'
lr = args['lr']  # 0.0001  # 0.0005  # 0.0004  # 0.001  # 0.025
decay = args['decay']  # 1e-6  #
use_initializer = args['use_initializer']
# add_noise = args['add_noise']
internal_noise = args['internal_noise']
skip_test = args['skip_test']
save_images = args['save_images']
save_predictions = args['save_predictions']
seed = args['seed']  # 420420420
trial = args['trial']
label = args['label']
project_dir = args['project_dir']
verbose = args['verbose']

assert 0 < trial

if verbose:
    pprint(args)

# Stimuli metadata
# luminance_weights = np.array([0.299, 0.587, 0.114])  # RGB (ITU-R 601-2 luma transform)
# data_set = 'CIFAR10'
# n_classes = 10
# classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
#            'dog', 'frog', 'horse', 'ship', 'truck')
# CIFAR10 image statistics calculated across the training set (after converting to greyscale)
# mean = 122.61930353949222
# std = 60.99213660091195
# colour = 'grayscale'  # 'rgb'
# contrast_level = 1  # Proportion of original contrast level for uniform and salt and pepper noise

weights = None  # Default unless pretrain flag is set

if convolution.capitalize() == 'Gabor':
    # Gabor parameters
    params = {# 'ksize': (127, 127), 
              'ksize': (63, 63),
              'gammas': [0.5], 
    #           'bs': np.linspace(0.4, 2.6, num=3),  # 0.4, 1, 1.8, 2.6
    #           'bs': np.linspace(0.4, 2.6, num=5),
              'bs': np.linspace(1, 2.6, num=3).tolist(),
    #           'bs': np.linspace(1, 2.6, num=5),
    #           'sigmas': [4, 8, 16],  # , 32 
              'sigmas': [8],
              'thetas': np.linspace(0, np.pi, 4, endpoint=False).tolist(),
              'psis': [np.pi/2, 3*np.pi/2]}
    filter_params = {convolution.capitalize(): params}
    mod = f'Gabor_{base}'
elif convolution == 'DoG':
    params = {'ksize': (63, 63),
              'sigmas': [1, 2, 4, 8],
#               'gammas': [2]
              'gammas': [1.6, 1.8, 2, 2.2]
             }
    filter_params = {convolution.capitalize(): params}
    mod = f'DoG_{base}'
elif convolution.capitalize() == 'Combined-full':
    params = {
        'DoG': {
            'ksize': (63, 63),
            'sigmas': [1, 2, 4, 8],
            'gammas': [1.6, 1.8, 2, 2.2]
            },
        'Gabor': {
            'ksize': (63, 63),  # TODO: Should this be reduced to reduce combined model size?
            'sigmas': [8],
            'gammas': [0.5], 
            'bs': np.linspace(1, 2.6, num=3).tolist(),
            'thetas': np.linspace(0, np.pi, 4, endpoint=False).tolist(),
            'psis': [np.pi/2, 3*np.pi/2]
            },
        }
    filter_params = params
    mod = f"{'+'.join(list(params))}_{base}"
    # mod = f'DoG+Gabor_{base}'
elif convolution.capitalize() == 'Combined-small':
    params = {
        'DoG': {
            'ksize': (15, 15),
            'sigmas': [1, 2, 4],
            'gammas': [1.6, 1.8, 2, 2.2]
            },
        'Gabor': {
            'ksize': (31, 31),
            'sigmas': [4],
            'gammas': [0.5], 
            'bs': np.linspace(1, 2.6, num=3).tolist(),
            'thetas': np.linspace(0, np.pi, 4, endpoint=False).tolist(),
            'psis': [np.pi/2, 3*np.pi/2]
            },
        }
    filter_params = params
    mod = f"{'+'.join(list(params))}_{base}"
    # mod = f'DoG+Gabor_{base}'
    # mod = f'{convolution}_{base}'
elif convolution.capitalize() == 'Combined-medium':
    params = {
        'DoG': {
            'ksize': (15, 15),
            'sigmas': [1, 2, 4, 8],
            'gammas': [1.6, 1.8, 2, 2.2]
            },
        'Gabor': {
            'ksize': (31, 31),
            'sigmas': [8],
            'gammas': [0.5], 
            'bs': np.linspace(1, 2.6, num=3).tolist(),
            'thetas': np.linspace(0, np.pi, 4, endpoint=False).tolist(),
            'psis': [np.pi/2, 3*np.pi/2]
            },
        }
    filter_params = params
    mod = f"{'+'.join(list(params))}_{base}"
elif convolution.capitalize() == 'Combined-trim':
    params = {
        'DoG': {
            'ksize': (31, 31),
            'sigmas': [1, 2, 4, 8],
            'gammas': [1.6, 1.8, 2, 2.2]
            },
        'Gabor': {
            'ksize': (31, 31),
            'sigmas': [8],
            'gammas': [0.5], 
            'bs': np.linspace(1, 2.6, num=3).tolist(),
            'thetas': np.linspace(0, np.pi, 4, endpoint=False).tolist(),
            'psis': [np.pi/2, 3*np.pi/2]
            },
        }
    filter_params = params
    # mod = f"{'+'.join(list(params))}_{base}"
    mod = f"Combined_{base}"
elif convolution.capitalize() == 'Low-pass':
    params = {'ksize': (63, 63),
#               'sigmas': [8]
              'sigmas': [1, 2, 4, 8]}  # long_k4
#               'sigmas': [2, 4]} # Low-pass_s_2_4
#               'sigmas': [4, 8]} # Low-pass_s_4_8
    filter_params = {convolution.capitalize(): params}
    mod = f'Low-pass_{base}'
elif convolution.capitalize() == 'Original':
#     params = None
    filter_params = params = None
#     mod = base
    mod = f"Original_{base}"
    if args['pretrain']:
        weights = 'imagenet'
        mod = f'{mod}_ImageNet'
else:
    warnings.warn(f'Unknown convolution type: {convolution}!')
    sys.exit()

# filter_params = params


# max_queue_size = 10
# workers = 12  # 4
# use_multiprocessing = False
# verbose = False
# report = 'batch'  # 'epoch'
# use_initializer = False
# extension = 'h5'  # For saving model/weights

# Setup project directories
if not project_dir:
    project_dir = os.path.dirname(os.path.realpath(__file__))
print(project_dir)

data_dir = os.path.join(project_dir, "data")
models_dir = os.path.join(project_dir, "models")
logs_dir = os.path.join(project_dir, "logs")
results_dir = os.path.join(project_dir, "results")

# data_dir = '/work/data'
# # Output paths
# models_dir = '/work/models'
# logs_dir = '/work/logs'
# results_dir = '/work/results'
os.makedirs(models_dir, exist_ok=True)
sim_results_dir = os.path.join(results_dir, label)  # label ignored if empty
os.makedirs(os.path.join(sim_results_dir, "metrics"), exist_ok=True)

if save_predictions:
    os.makedirs(os.path.join(sim_results_dir, 'predictions'), exist_ok=True)

if save_images:
    image_out_dir = os.path.join(sim_results_dir, 'img')
    os.makedirs(image_out_dir, exist_ok=True)
else:
    image_out_dir = None
    image_prefix = ''

print('=' * 80)


# Hardcode noise levels
n_levels = 11
noise_types = get_perturbations(n_levels=n_levels)
# noise_types = [("Uniform", uniform_noise, np.linspace(0, 1, n_levels)),
#                ("Salt and Pepper", salt_and_pepper_noise, np.linspace(0, 1, n_levels)),
#             #    ("High Pass", high_pass_filter, np.logspace(np.log10(5), np.log10(0.3), n_levels)),
#                ("High Pass", high_pass_filter, np.logspace(2, 0, n_levels)),
#             #    ("Low Pass", low_pass_filter, np.logspace(0, np.log10(40), n_levels)),
#                ("Low Pass", low_pass_filter, np.logspace(0, 2, n_levels)),
#                ("Contrast", adjust_contrast, np.logspace(0, -2, n_levels)),
#                ("Phase Scrambling", scramble_phases, np.linspace(0, 180, n_levels)),
#                ("Darken", adjust_brightness, np.linspace(0, -1, n_levels)),
#                ("Brighten", adjust_brightness, np.linspace(0, 1, n_levels)),
#                ("Rotation", rotate_image, np.array([0, 90, 180, 270], dtype=int)),
#                ('Invert', invert_luminance, np.array([0, 1], dtype=int))]

# # Process stimuli
# if upscale:
#     image_size = (224, 224)
#     image_shape = image_size + (1,)
#     # image_shape = (224, 224, 1)
# else:
#     image_size = (32, 32)
#     image_shape = image_size + (1,)
#     # image_shape = (32, 32, 1)

# interpolation = cv2.INTER_LANCZOS4  # cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC
interpolation_name = interpolation_names[interpolation]

preload = False  # NOTE: Disabled as this should only be used with small image sets

if image_path and os.path.isdir(image_path):
    data_set = os.path.basename(os.path.normpath(image_path))
    load_images_from_disk = True
    # NOTE: Expects `image_path` to have sub-directories "train" and "test"

    # NOTE: Assumes the same classes as CIFAR10 (imported from config)
    assert os.path.isdir(os.path.join(image_path, 'train'))
    if os.path.isdir(os.path.join(image_path, 'val')):
        validation_image_path = os.path.join(image_path, 'val')
    else:
        assert os.path.isdir(os.path.join(image_path, 'test'))
        validation_image_path = os.path.join(image_path, 'test')
    perturbation_image_path = validation_image_path  # NOTE: This uses validation images if present, otherwise the test set

    if preload:
        train_image_sets, x_train, y_train = utils.load_images(os.path.join(image_path, 'train'))
        # Overwrite classes
        classes = list(train_image_sets)
        n_classes = len(classes)  # len(train_image_sets)

        if os.path.isdir(validation_image_path):
            val_image_sets, x_val, y_val = utils.load_images(validation_image_path)
            assert n_classes == len(val_image_sets)

        test_image_sets, x_test, y_test = utils.load_images(os.path.join(image_path, 'test'), shuffle=False)
        assert n_classes == len(test_image_sets)
        test_images_path = ""
    else:
        x_test = None
        y_test = None
        test_images_path = perturbation_image_path

# if test_image_path and os.path.isdir(test_image_path):

else:  # Default to standard CIFAR10 training and testing images

    load_images_from_disk = False
    test_images_path = ""
    # Set up stimuli
    data_set = "CIFAR10"
    # Imported from config
#     classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
#                'dog', 'frog', 'horse', 'ship', 'truck')
#     n_classes = len(classes)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # RGB format
    x_train = np.expand_dims(np.dot(x_train, luminance_weights), axis=-1)
    x_test = np.expand_dims(np.dot(x_test, luminance_weights), axis=-1)
    y_train = to_categorical(y_train, num_classes=n_classes, dtype='uint8')
    y_test = to_categorical(y_test, num_classes=n_classes, dtype='uint8')

# print('-' * 80)
if not load_images_from_disk or preload:
    if upscale:
        if interpolation:  #interpolate:
            print(f'Interpolating upscaled images with "{interpolation}" ({interpolation_name})...')
            x_train_interp = np.zeros(shape=(x_train.shape[0], *image_shape), dtype=np.float16)
            for i, image in enumerate(x_train):
                x_train_interp[i, :, :, 0] = cv2.resize(image, dsize=image_size, 
                                                    interpolation=interpolation)
            del x_train
            x_train = x_train_interp
            x_train[x_train < 0] = 0
            x_train[x_train > 255] = 255

            x_test_interp = np.zeros(shape=(x_test.shape[0], *image_shape), dtype=np.float16)
            for i, image in enumerate(x_test):
                x_test_interp[i, :, :, 0] = cv2.resize(image, dsize=image_size, 
                                                    interpolation=interpolation)
            del x_test
            x_test = x_test_interp        
            x_test[x_test < 0] = 0
            x_test[x_test > 255] = 255
        else:  # Redundant as this is equivalent to interpolation == 0
            # Equivalent to cv2.INTER_NEAREST (or PIL.Image.NEAREST)
            x_train = x_train.repeat(7, axis=1).repeat(7, axis=2)
            x_test = x_test.repeat(7, axis=1).repeat(7, axis=2)

    # NOTE: This is later overridden by the ImageDataGenerator to tf.keras.backend.floatx() (default: 'float32')
    x_train = x_train.astype(np.float16)
    x_test = x_test.astype(np.float16)

    # TODO: Implement for testing stimuli
    # if weights == 'imagenet':
    #     print("Replicating grayscale layer to match expected input size...")
    #     x_train = x_train.repeat(3, axis=-1)
    #     x_test = x_test.repeat(3, axis=-1)

    # Summarise stimuli
    print(f'x_train.shape: {x_train.shape}')
    print(f'Training: {x_train.shape[0]} in {y_train.shape[1]} categories')
    print(f'Testing: {x_test.shape[0]} in {y_test.shape[1]} categories')

# if data_set.upper() == 'CIFAR10' and colour == 'grayscale':
#     if interpolation in train_image_stats:
#         mean, std = train_image_stats[interpolation]
# #     if (interpolation == cv2.INTER_NEAREST) or not interpolate:
# #         mean = 122.61930353949222
# #         std = 60.99213660091195
# #     elif interpolation == cv2.INTER_LANCZOS4:
# #         # Without clipping
# #         # mean = 122.6172103881836
# #         # std = 60.89457321166992
# #         # After clipping
# #         mean = 122.61385345458984
# #         std = 60.87860107421875
#     else:
#         print(f'Uncached interpolation method: {interpolation}')
#         recalculate_statistics = True
# else:
#     recalculate_statistics = True

# Default normalisation method
featurewise_normalisation = True
samplewise_normalisation = False
        
if (not recalculate_statistics
    and colour == 'grayscale'
    and image_size == (224, 224)
    and data_set.lower() in train_image_stats
    and interpolation_name in train_image_stats[data_set.lower()]):
    
    mean, std = train_image_stats[data_set.lower()][interpolation_name]
else:  # Featurewise statistics not cached
    print(f'Uncached interpolation method: {interpolation_name} for {data_set}!')
    recalculate_statistics = True

    print('Recalculating training image statistics...')    
    if load_images_from_disk:
        featurewise_normalisation = False
        samplewise_normalisation = True
        mean = None
        std = None
        print("INFO: Loading images form disk so switching to samplewise normalisation.")
#         # Fit to a random sample of images from directory
#         gen_train_sample = data_gen.flow_from_directory(
#             os.path.join(image_path, 'train'),
#             target_size=image_size,
#             color_mode=colour,
#             interpolation='lanczos',
#             batch_size=1024,
#             shuffle=True,
#             seed=0,
#             save_to_dir=None,
#             follow_links=True,
#             subset=None
#         )
#         data_gen.fit(gen_train_sample)

#         mean = np.squeeze(data_gen.mean).tolist()
#         std = np.squeeze(data_gen.std).tolist()
    else:
        featurewise_normalisation = True
        samplewise_normalisation = False
        data_gen = ImageDataGenerator(featurewise_center=featurewise_normalisation,
                                      featurewise_std_normalization=featurewise_normalisation)
        data_gen.fit(x_train)

        # TODO: Calculate statistics from sample and unindent below
        mean = np.squeeze(data_gen.mean).tolist()
        std = np.squeeze(data_gen.std).tolist()
print(f'Training statistics: mean={mean}; std={std}')

# Save metadata
# TODO: Simplify by using the args dictionary
sim = {
    'data_set': data_set,
    'image_path': image_path,
    'n_classes': n_classes,
    'classes': classes,
    'train': train,
    'epochs': epochs,
    'optimizer': optimizer,
    'lr': lr,
    'decay': decay,
    'batch': batch,
    'data_augmentation': data_augmentation,
    'extra_augmentation': extra_augmentation,
    'seed': seed,
    'trial': trial,
    'model': mod,
    'convolution': convolution,
    'base': base,
    'weights': weights,
    'label': label,
    'noise': {noise: levels.tolist() for noise, _, levels in noise_types},
    'image_mean': mean,
    'image_std': std,
    'image_shape': image_shape,
    'upscale': upscale,
#     'interpolate': interpolate,
    'interpolation': interpolation,
    'recalculate_statistics': recalculate_statistics,
    'colour': colour,
    'luminance_rgb_weights': luminance_weights.tolist(),
    'contrast_level': contrast_level,
    'save_predictions': save_predictions,
    'image_out_dir': image_out_dir,
    'models_dir': models_dir,
    'results_dir': results_dir,
    'use_initializer': use_initializer,
#     'add_noise': add_noise,
    'internal_noise': internal_noise,
    'filter_params': params,
    }

# TODO: Replace with f'{conv}_{base}_{trial}'
model_name = f'{mod}_{trial}'
# # sim_set = f"test_{datetime.now().strftime('%Y%m%d')}"
# if label:  # len(label) > 0:
#     sim_set = f"{mod}_{label}_t{trial}_e{epochs}_s{seed}"
# else:
#     sim_set = f"{mod}_t{trial}_e{epochs}_s{seed}"
sim_set = f"{model_name}_s{seed}"
sim_file = f"{sim_set}.json"
os.makedirs(os.path.join(sim_results_dir, 'parameters'), exist_ok=True)
with open(os.path.join(sim_results_dir, "parameters", sim_file), "w") as sf:
    json.dump(sim, sf, indent=4)

if save_images:
    stimuli = {
        'noise': {noise: levels.tolist() for noise, _, levels in noise_types},
        'image_mean': mean,
        'image_std': std,
        'image_shape': image_shape,
        'colour': colour,
        'luminance_rgb_weights': luminance_weights.tolist(),
        'contrast_level': contrast_level,
        }
    with open(os.path.join(image_out_dir, 'stimuli.json'), "w") as sf:
        json.dump(stimuli, sf, indent=4)


# for trial in range(start_trial, n_seeds+1):
# seed = start_seed * trial
# for m, mod in enumerate(models):

print('=' * 80)  # Build/load model
print(f"Creating {model_name}...", flush=True)
# Create the model


# get_all_cnn = functools.partial(allcnn, image_shape=image_shape, n_classes=n_classes)

# @functools.wraps(allcnn)
def get_all_cnn(include_top=True, weights=None, input_shape=image_shape, classes=n_classes):
    # model = functools.partial(allcnn, image_shape=image_shape, n_classes=n_classes)
    return allcnn(image_shape=input_shape, n_classes=n_classes)
    # return allcnn_imagenet(image_shape=input_shape, n_classes=n_classes)

model_base = {'vgg16': tf.keras.applications.vgg16.VGG16, 
              'vgg19': tf.keras.applications.vgg19.VGG19,
              'resnet': BioResNet50,
#               'resnet': tf.keras.applications.resnet_v2.ResNet50V2,
              'mobilenet': tf.keras.applications.mobilenet_v2.MobileNetV2, # MobileNetV2
              'inception': tf.keras.applications.inception_v3.InceptionV3,
              'allcnn': get_all_cnn}

# ResNet50, Inception V3, and Xception

# input_tensor = Input(shape=image_shape, name='input_1', dtype='float16')
if weights is None:
    output_classes = n_classes
else:
    output_classes = 1000  # Default
base_name = base.lower().replace('-', '')

if base_name not in ["resnet"]:  # List of hard-coded exceptions
    model = model_base[base_name](include_top=True, 
                              weights=weights, 
                            #   input_tensor=input_tensor,
                              input_shape=image_shape,
                              classes=output_classes)
    # if add_noise:
    #     model = utils.insert_noise_layer(model, layer=None, std=noise)
    model = utils.substitute_layer(model, params,
                                   filter_type=convolution,
                                   replace_layer=None,
                                   input_shape=image_size,
                                   colour_input=colour,
                                   use_initializer=use_initializer,
                                   noise_std=internal_noise)
else:
    model = model_base[base_name](include_top=True, 
                                  weights=weights,
                                  kernels=filter_params,
                                  # input_tensor=input_tensor,
                                  input_shape=image_shape,
                                  classes=output_classes)
if n_classes != output_classes:  # 1000:
    model = utils.substitute_output(model, n_classes=n_classes)

opt_args = {'lr': lr, 'decay': decay}
# if optimizer in []:

opt = tf.keras.optimizers.get({'class_name': optimizer, 'config': opt_args})
# opt = tf.keras.optimizers.RMSprop(lr=lr, decay=1e-6)  # CIFAR10

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
if verbose:
    model.summary()

# TODO: Move to new SavedModel format
# https://www.tensorflow.org/tutorials/keras/save_and_load#savedmodel_format
model_output_dir = os.path.join(models_dir, label, model_name)
os.makedirs(model_output_dir, exist_ok=True)
full_path_to_model = os.path.join(model_output_dir, f"{epochs:03d}_epochs")

print(f"Trial: {trial}; seed={seed}", flush=True)

model_data_file = f"{full_path_to_model}_weights.{extension}"

if not train:
    print(f"Loading {model_name}...", flush=True)
    model.load_weights(model_data_file)
    print(f"{model_name} loaded!", flush=True)
else:
    # Create Image Data Generators
    if data_augmentation:
        if extra_augmentation:
            print('Using extra data augmentation.')
            data_gen = ImageDataGenerator(
                featurewise_center=featurewise_normalisation,  # True,
                featurewise_std_normalization=featurewise_normalisation,  # True,
                samplewise_center=samplewise_normalisation,  # False,
                samplewise_std_normalization=samplewise_normalisation,  # False,
                zca_whitening=False,
                rotation_range=45,
                brightness_range=(0.2, 1.0),
                shear_range=0.2,
                zoom_range=(0.5, 1.5),
    #             fill_mode="constant",
    #             cval=mean,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False)
        else:
            print('Using data augmentation.')
            data_gen = ImageDataGenerator(
                featurewise_center=featurewise_normalisation,  # True,
                featurewise_std_normalization=featurewise_normalisation,  # True,
                samplewise_center=samplewise_normalisation,  # False,
                samplewise_std_normalization=samplewise_normalisation,  # False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False)
    else:
        data_gen = ImageDataGenerator(
            #preprocessing_function=prep_image,
            featurewise_center=featurewise_normalisation,
            featurewise_std_normalization=featurewise_normalisation,
            samplewise_center=samplewise_normalisation,
            samplewise_std_normalization=samplewise_normalisation,
        )
    # data_gen.fit(x_train)
    if featurewise_normalisation:
        data_gen.mean = mean
        data_gen.std = std

    # NOTE: When classes=None, the dictionary containing the mapping from class names to class indices can be obtained via the attribute class_indices.

    if load_images_from_disk:
        gen_train = data_gen.flow_from_directory(
            os.path.join(image_path, 'train'),
            target_size=image_size,
            color_mode=colour,
#             classes=classes,
#             class_mode='categorical',
            batch_size=batch,
            shuffle=True,
            seed=seed,
            save_to_dir=None,
            follow_links=True,
            interpolation=interpolation_names[interpolation],
            subset=None
        )
        #assert gen_train.class_indices
        gen_valid = data_gen.flow_from_directory(
            validation_image_path,
            target_size=image_size,
            color_mode=colour,
            batch_size=batch,
            shuffle=True,
            seed=seed,
            save_to_dir=None,
            follow_links=True,
            interpolation=interpolation_names[interpolation],
            subset=None
        )
    else:
        gen_train = data_gen.flow(x_train, y=y_train, batch_size=batch, 
                                    shuffle=True, seed=seed, save_to_dir=None)
        gen_valid = data_gen.flow(x_test, y=y_test, batch_size=batch, 
                                    shuffle=True, seed=seed, save_to_dir=None)

    print(f'Checking for {model_data_file}...', flush=True)
    if os.path.exists(model_data_file) and not clean:
        print(f"Found {mod} - skipping training...", flush=True)
        model.load_weights(model_data_file)  # TODO: Check load_weights works when the whole model is saved
        print(f"{model_name} loaded!", flush=True)
    else:
        print(f"Training {mod} for {epochs} epochs...", flush=True)
        t0 = time.time()

        callbacks = []
        if args['log']:
            # Create a tensorboard callback
            # logdir = '/work/logs/scalars/' + datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = os.path.join(logs_dir, 'scalars', f'{model_name}-{datetime.now():%Y%m%d-%H%M%S}')
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=5, update_freq='epoch')  # 2048)
            callbacks.append(tensorboard_cb)

        resume_training = False
        csv_logger_cb = tf.keras.callbacks.CSVLogger(os.path.join(logs_dir, f'{model_name}.csv'), 
                                                        append=resume_training, separator=',')
        callbacks.append(csv_logger_cb)

        # Create a callback that saves the model's weights
        checkpoint_path = os.path.join(models_dir, "model.ckpt")  # f"{model_name}.ckpt"
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                           save_weights_only=True,
                                                           verbose=0)
        callbacks.append(checkpoint_cb)

        save_freq = None  # 10
        if save_freq:
            weights_path = os.path.join(model_output_dir, "{epoch:03d}_epochs.h5")
            weights_cb = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path,
                                                            save_weights_only=True,
                                                            verbose=1, period=save_freq)
            callbacks.append(weights_cb)

        reduce_lr_on_plateau = True
        if reduce_lr_on_plateau:
            reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                                patience=5, min_lr=1e-8, verbose=1)
            callbacks.append(reduce_lr_cb)

        # Alternative from Geirhos et al. 
        # Set training schedule
        # assert len(boundaries) <= 4
        # boundaries = [-1] * (4 - len(boundaries)) + boundaries
        # print('epoch boundaries for finetuning: {}'.format(boundaries))
        # boundaries = [pretrained_epochs_new + x for x in boundaries]
        # decay_rates = [1, 0.1, 0.01, 0.001, 1e-4]

        # def scheduler(epoch):
        #     if epoch < 10:
        #         return 0.001
        #     else:
        #         return 0.001 * tf.math.exp(0.1 * (10 - epoch))
        #         # return lr * 0.5 ** (epoch // 10)
        # def lr_scheduler(epoch):
        #     return learning_rate * (0.5 ** (epoch // lr_drop))

        # scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

        # history = model.fit_generator(gen_train,
        #                             #   steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch))),
        #                               steps_per_epoch=gen_train.n//batch,
        #                               epochs=epochs,
        #                               validation_data=gen_valid,
        #                               validation_steps=gen_valid.n//batch,
        #                               shuffle=True,
        #                               callbacks=callbacks,
        #                               max_queue_size=max_queue_size,
        #                               use_multiprocessing=use_multiprocessing,
        #                               workers=workers)

        # if resume_training:
        #     initial_epoch = ...
        # else:
        #     initial_epoch = 0
        history = model.fit(gen_train,
                            epochs=epochs,
                            # steps_per_epoch and steps_per_epoch are required due to a regression in TF 2.2
                            # https://github.com/tensorflow/tensorflow/issues/37968
                            # steps_per_epoch=gen_train.n//batch,
                            steps_per_epoch=len(gen_train),
                            callbacks=callbacks,
                            validation_data=gen_valid,
                            # validation_steps=gen_valid.n//batch,
                            validation_steps=len(gen_valid),
                            shuffle=True,
                            max_queue_size=max_queue_size,
                            workers=workers,
                            use_multiprocessing=use_multiprocessing)

        if use_initializer:
            model.save_weights(f"{full_path_to_model}_weights.{extension}")  # weights only
            # Does not work with lambda layer
            with open(f"{full_path_to_model}.json", "w") as sf:
                sf.write(model.to_json())  # architecture only
        else:
            model.save(f"{full_path_to_model}.{extension}")  # Full model
        with open(os.path.join(model_output_dir, "simulation.json"), "w") as sf:
            json.dump(sim, sf, indent=4)

        learning_curves = os.path.join(logs_dir, f'{model_name}.png')  # f'{mod}_train_CIFAR10_{trial}.png')
        plots.plot_history(history, chance=1/n_classes, filename=learning_curves)

        t_elapsed = time.time() - t0
        print(f"{model_name} training finished [{str(timedelta(seconds=t_elapsed))}]!", flush=True)
print("=" * 80)

if skip_test:
    print("Skipping testing.", flush=True)
    tf.keras.backend.clear_session()  # Clear GPU memory
    print("=" * 80)
    sys.exit()

# batch = 100  # Differences in metrics may be due to rounding effects


# Test Generalisation Images

if isinstance(test_generalisation, str):
    if test_generalisation.lower() == 'all':
        test_sets = generalisation_types
    elif test_generalisation.lower() in generalisation_types:
        test_sets = [test_generalisation.lower()]
    else:
        warnings.warn(f'Unknown generalisation test set: {test_generalisation}!')
        test_sets = []
elif isinstance(test_generalisation, bool):
    if test_generalisation:
        test_sets = generalisation_types
    else:
        test_sets = []
else:
    warnings.warn(f'Unknown generalisation test set type: {test_generalisation} ({type(test_generalisation)})!')
    test_sets = []

# if save_predictions:
#     frames = []

if test_generalisation:
    if invert_test_images:
        # test_sets.extend([f'{test_set}_inverted' for test_set in test_sets])
        inversions = [False, True]
    else:
        inversions = [False]

    results_file = os.path.join(sim_results_dir, "metrics", f"{model_name}_generalise_s{seed}.csv")
    with open(results_file, 'w') as results:
        writer = csv.DictWriter(results, fieldnames=generalisation_columns)
        writer.writeheader()

# if test_image_path and os.path.isdir(test_image_path):
for test_set in test_sets:
    generalisation_image_path = os.path.join(data_dir, "CIFAR-10G", f"{image_size[0]}x{image_size[1]}", test_set)
    assert os.path.isdir(generalisation_image_path)

    for invert in inversions:
        print(f"Testing {model_name} with images from {generalisation_image_path}{' (inverted)' if invert else ''}...", flush=True)
        t0 = time.time()
        # rng = np.random.RandomState(seed=seed)

        full_set_name = f"{test_set}{'_inverted' if invert else ''}"
        # NOTE: Generalisation test images are already in [0, 1] so do not rescale before preprocessing
#         if test_set in ['scharr']:
#             rescale = 1/255
#         else:
#             rescale = 1/255  # 1
        rescale = 1/255

        # Old method: create inverted images on the fly
#         if invert:
#             # prep_image = cifar_wrapper(functools.partial(invert_luminance, level=1),
#             #                            rescale=rescale)
#             prep_image = get_noise_preprocessor("Invert", invert_luminance, level=1, rescale=rescale)
#         else:
#             # prep_image = cifar_wrapper(sanity_check, rescale=rescale)
#             prep_image = get_noise_preprocessor("None", rescale=rescale)

        # New method: use the same preprocessor and load pre-inverted images
        prep_image = get_noise_preprocessor("None", rescale=rescale)  # The default is rescale=1/255
        if invert:
            generalisation_image_path = f"{generalisation_image_path}_inverted"
            assert os.path.isdir(generalisation_image_path)

        data_gen = ImageDataGenerator(# rescale=255,
                                      preprocessing_function=prep_image,
                                      featurewise_center=True, 
                                      featurewise_std_normalization=True)

        # data_gen.fit(x_train)  # Set mean and std
#         if invert:
#             data_gen.mean = 255 - mean
#             data_gen.std = std
#         else:
#             data_gen.mean = mean
#             data_gen.std = std
        data_gen.mean = mean
        data_gen.std = std

        if save_images:
            generalisation_image_out_dir = os.path.join(image_out_dir, full_set_name)
            os.makedirs(generalisation_image_out_dir, exist_ok=True)
            generalisation_prefix = ''
        else:
            generalisation_image_out_dir = None
            generalisation_prefix = ''

        gen_test = data_gen.flow_from_directory(generalisation_image_path,
                                                target_size=image_size,
                                                color_mode=colour,
                                                batch_size=batch,
                                                shuffle=False, seed=seed,
                                                interpolation=interpolation_names[interpolation],
                                                save_to_dir=generalisation_image_out_dir, 
                                                save_prefix=generalisation_prefix)

        metrics = model.evaluate(gen_test, 
                                 steps=len(gen_test),
                                 verbose=1,
                                 max_queue_size=max_queue_size,
                                 workers=workers,
                                 use_multiprocessing=use_multiprocessing)


        if train:
            metrics_dict = {metric: score for metric, score in zip(model.metrics_names, metrics)}
            print(f"Evaluation results: {metrics_dict}")
        else:
            print(f"Evaluation results: {metrics}")

        if save_predictions:  # Get classification probabilities
            # Reinitialise iterator
            gen_test = data_gen.flow_from_directory(generalisation_image_path,
                                        target_size=image_size,
                                        color_mode=colour,
                                        batch_size=batch,
                                        shuffle=False, seed=seed,
                                        interpolation=interpolation_names[interpolation],
                                        save_to_dir=generalisation_image_out_dir, 
                                        save_prefix=generalisation_prefix)

            predictions = model.predict(gen_test, 
                                        verbose=1,
                                        # steps=gen_test.n//batch,  # BAD: This skips the remainder of images
                                        steps=len(gen_test),
                                        max_queue_size=max_queue_size,
                                        workers=workers,
                                        use_multiprocessing=use_multiprocessing)
            # print(predictions.shape)  # (n_images, n_classes)
            file_name = f"{model_name}_generalise_{full_set_name}_s{seed}.csv"
            predictions_file = os.path.join(sim_results_dir, 'predictions', file_name)

#             np.savetxt(predictions_file, predictions, delimiter=',', 
#                        header=','.join([f'p(class={c})' for c in classes]))

            n_images_per_class = 10
            n_images = n_images_per_class * n_classes
            y_generalise = np.repeat(range(n_classes), n_images_per_class)
            classifications = np.argmax(predictions, axis=1)

#             loss = categorical_crossentropy(to_categorical(y_generalise, num_classes=n_classes, dtype='uint8'), predictions)
#             loss = np.sum(loss.numpy())
#             assert np.isclose(loss, metrics[0]), f"Loss: Calculated: {loss} =/= Recorded: {metrics[0]}"
            # Check accuracy based on probabilities matches accuracy from .evaluate
            assert len(classifications) == len(y_generalise)
            accuracy = sum(classifications == y_generalise) / len(y_generalise)
#             cat_acc = categorical_accuracy(y_generalise, classifications).numpy()
#             assert np.isclose(accuracy, cat_acc), f"Calculated: {accuracy} =/= Library: {cat_acc}"
            assert np.isclose(accuracy, metrics[1], atol=0.001), f"Calculated: {accuracy} =/= Evaluated: {metrics[1]}"
#             if not np.isclose(accuracy, metrics[1], atol=0.001):  # 1/(2*n_images)
#                 print(f"Calculated: {accuracy} =/= Evaluated: {metrics[1]}")

            # Put predictions into DataFrame
            df_gen = pd.DataFrame(predictions, columns=classes)
            df_gen["Predicted"] = classifications
            df_gen["Class"] = y_generalise
            df_gen["Correct"] = classifications == y_generalise
            df_gen["Image"] = range(n_images)
            df_gen["Set"] = [full_set_name] * n_images
            df_gen["Type"] = [test_set] * n_images
            df_gen["Inverted"] = [invert] * n_images

#             frames.append(df_gen)
            df_gen.to_csv(predictions_file, index=False)

            if verbose:
                print(f'Predictions written to: {predictions_file}')
            del predictions

        if save_predictions:
            # Manual calculation is more accurate, probably due to rounding errors
            # However, the results are the same for dtype=float16 rounded to 7 d.p.
            acc = accuracy
        else:
            acc = metrics[1]
        # generalisation_columns = ['Model', 'Convolution', 'Base', 'Weights', 'Trial', 'Seed',
        #                           'Set', 'Type', 'Inverted', 'Loss', 'Accuracy']

        row = {'Model': mod, 'Convolution': convolution, 'Base': base,
               'Weights': str(weights), 'Trial': trial, 'Seed': seed,
               'Set': full_set_name, 'Type': test_set, 'Inverted': invert,
               'Loss': metrics[0], 'Accuracy': acc}
        with open(results_file, 'a') as results:
            writer = csv.DictWriter(results, fieldnames=generalisation_columns)
            writer.writerow(row)

        t_elapsed = time.time() - t0
        print(f"Testing {test_set}{' (inverted)' if invert else ''} images finished! [{t_elapsed:.3f}s]", flush=True)
        print("-" * 80)

        # Clean up
        del data_gen
        gc.collect()

    print('Generalisation testing finished!')
if not len(test_sets):
    print('Generalisation testing skipped!')
print("=" * 80)

# Clear GPU memory
# tf.keras.backend.clear_session()
# sys.exit()


if not test_perturbations:
    # Clear GPU memory
    tf.keras.backend.clear_session()
    sys.exit()


ver_num = [int(x, 10) for x in tf.__version__.split('.')]
if (ver_num[0] >= 2) and (ver_num[1] > 2) and True:
    print("Setting default dtype to float16")
    tf.keras.backend.set_floatx('float16')  # Set default dtype to 16 bit


# Test perturbation images

# Test on perturbed test images
test_set = data_set
results_file = os.path.join(sim_results_dir, "metrics", f"{model_name}_perturb_{test_set.lower()}_s{seed}.csv")
if test_perturbations and (not os.path.isfile(results_file) or clean):
    test_noise_perturbations(model, sim, noise_types, sim_results_dir=sim_results_dir, 
                             test_set=test_set, test_images_path=test_images_path, x_test=x_test, y_test=y_test)

    
# Test on perturbed generalisation images
for test_set in generalisation_sets:
    test_images_path = os.path.join(data_dir, "CIFAR-10G", f"{image_size[0]}x{image_size[1]}", test_set)
    assert os.path.isdir(test_images_path)

    results_file = os.path.join(sim_results_dir, "metrics", f"{model_name}_perturb_{test_set.lower()}_s{seed}.csv")
    if test_perturbations and (not os.path.isfile(results_file) or clean):
        test_noise_perturbations(model, sim, noise_types, sim_results_dir=sim_results_dir,
                                 test_set=test_set, test_images_path=test_images_path)


print(f'Models: {model_output_dir}')
print(f'Logs: {logs_dir}')  # if args['log']: logdir
print(f'Results: {sim_results_dir}')
if save_predictions:
    print(f"Predictions: {os.path.join(sim_results_dir, 'predictions')}")
if save_images:
    print(f'Generated images: {image_out_dir}')

print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Simulation for model "{model_output_dir}" finished!')
print("=" * 80)
# for m, metric in enumerate(model.metric_names):
#     test_metrics[mod][metric]

# results = pd.DataFrame(rows)

# Clear GPU memory
tf.keras.backend.clear_session()

# return model, results
