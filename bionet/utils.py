"""
This module contains help functions and general utilities for building custom convolutional layers.
"""

import os
import json
import glob
# import subprocess
from pprint import pprint

import pandas as pd
import numpy as np
# from scipy import signal
import cv2
# from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Conv2D  # , Dense
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.datasets import cifar10
from tensorflow.python.framework import dtypes
from tensorflow.keras.utils import to_categorical
# from K.tensorflow_backend import set_session
# from K.tensorflow_backend import clear_session
# from K.tensorflow_backend import get_session
from sklearn.metrics import auc
from scipy.integrate import simps

# Needed for load_images
from tqdm import tqdm
from matplotlib import pyplot as plt

from bionet.config import (luminance_weights, generalisation_sets, classes)
from bionet.preparation import get_perturbations
# all_test_sets = ['line_drawings', 'silhouettes', 'contours']  # , 'scharr']
# generalisation_sets = ['line_drawings', 'silhouettes', 'contours',
#                        'line_drawings_inverted', 'silhouettes_inverted', 'contours_inverted']

# # Reset Keras Session
# def reset_keras():
#     sess = get_session()
#     clear_session()
#     sess.close()
#     sess = get_session()

#     try:
#         del classifier # this is from global space - change this as you need
#     except:
#         pass

#     print(gc.collect()) # if it's done something you should see a number being outputted

#     # use the same config as you used to create the session
#     config = tensorflow.ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = 1
#     config.gpu_options.visible_device_list = "0"
#     set_session(tensorflow.Session(config=config))



def get_simulation_params(sim_set, model_name, models_dir='/work/models'):
    with open(os.path.join(models_dir, sim_set, model_name, 'simulation.json'), 'r') as fh:
        sim = json.load(fh)
    return sim


def get_cifar10_images(colour='grayscale', upscale=True, interpolate=True, fresh=False):

    # TODO: Save the processed image arrays to disk and reload if not fresh
    
    n_classes = len(classes)
    interpolation = cv2.INTER_LANCZOS4  # cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC

    # Process stimuli
    if upscale:
        image_size = (224, 224)
        image_shape = image_size + (1,)
        # image_shape = (224, 224, 1)
    else:
        image_size = (32, 32)
        image_shape = image_size + (1,)
        # image_shape = (32, 32, 1)

    # Set up stimuli
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # RGB format
    x_train = np.expand_dims(np.dot(x_train, luminance_weights), axis=-1)
    x_test = np.expand_dims(np.dot(x_test, luminance_weights), axis=-1)
    y_train = to_categorical(y_train, num_classes=n_classes, dtype='uint8')
    y_test = to_categorical(y_test, num_classes=n_classes, dtype='uint8')

    if upscale:
        if interpolate:
            print(f'Interpolating upscaled images with "{interpolation}"...')
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
        else:
            # Equivalent to cv2.INTER_NEAREST (or PIL.Image.NEAREST)
            x_train = x_train.repeat(7, axis=1).repeat(7, axis=2)
            x_test = x_test.repeat(7, axis=1).repeat(7, axis=2)

    # NOTE: This is later overridden by the ImageDataGenerator which has 'float32' as the default
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

    if colour == 'grayscale':
        if (interpolation == cv2.INTER_NEAREST) or not interpolate:
            mean = 122.61930353949222
            std = 60.99213660091195
        elif interpolation == cv2.INTER_LANCZOS4:  # After clipping
            mean = 122.61385345458984
            std = 60.87860107421875
        else:
            print(f'Uncached interpolation method: {interpolation}')
            recalculate_statistics = True
    else:
        recalculate_statistics = True
    
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std

    return (x_train, y_train), (x_test, y_test)


def get_images(image, preprocessing_function=None, image_dir='/work/data'):

    if isinstance(image, np.ndarray):  # Passed image array
        if preprocessing_function:
            return preprocessing_function(image)
        else:
            return image
    elif isinstance(image, str):
        if os.path.isfile(image):  # Passed a file path
            if preprocessing_function:
                return preprocessing_function(plt.imread(image))
            else:
                return plt.imread(image)
        if image is None or image == 'all':  # Process generalisation image sets
            image_sets = generalisation_sets
        else:
            assert image in generalisation_sets
            image_sets = [image]

    images = {}

    for image_set in image_sets:
        images[image_set] = {}
        image_set_dir = os.path.join(image_dir, image_set)
        for c_ind, category in enumerate(sorted(os.listdir(image_set_dir))):
            images[image_set][category] = []
            for s_ind, image_file in enumerate(sorted(os.listdir(os.path.join(image_set_dir, category)))):

                full_path = os.path.join(image_set_dir, category, image_file)

                image = plt.imread(full_path) * 255  # Generalisation images are in the range [0, 1]

                if image.shape == (224, 224):
                    image = image[..., np.newaxis]
                if image.shape == (224, 224, 3):
                    image = np.dot(image, luminance_weights)[..., np.newaxis]
                # image = np.squeeze(prep_image(image))
                if preprocessing_function:
                    image = preprocessing_function(image)

                image[image < 0] = 0
                image[image > 255] = 255

                # image -= mean
                # image /= std

                images[image_set][category].append(image)

    return images


def load_images(path, shuffle=True, verbose=1):

    image_set = {}
    for root, dirs, files in os.walk(path, followlinks=True):
        if root == path:
            categories = sorted(dirs)
            image_set = {cat: [] for cat in categories}
        else:
            image_set[os.path.basename(root)] = sorted(files)

    n_cat_images = {cat: len(files) for (cat, files) in image_set.items()}
    n_images = sum(n_cat_images.values())
    image_dims = plt.imread(os.path.join(path, categories[0],
                            image_set[categories[0]][0])).shape

    disable = True
    if verbose:
        print(image_dims)
        disable = False
    X = np.zeros((n_images, image_dims[0], image_dims[1], 1), dtype='float16')  # 'float32'
    y = np.zeros((n_images, len(categories)), dtype=int)

    tally = 0
    for c, (cat, files) in enumerate(tqdm(image_set.items(), desc=path, disable=disable)):
        for i, image in enumerate(files):
            cimg = cv2.imread(os.path.join(path, cat, image))  # cv2 opens in BGR
            X[i+tally] = np.expand_dims(cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY), axis=-1)
            # Alternative in one operation
            # X[i+tally] = np.expand_dims(cv2.imread(os.path.join(path, cat, image), 
            #                                        cv2.IMREAD_GRAYSCALE), axis=-1)
        y[tally:tally+len(files), c] = True
        tally += len(files)

    if not shuffle:
        return image_set, X, y

    shuffle = np.random.permutation(y.shape[0])

    return image_set, X[shuffle], y[shuffle]


def load_predictions(data_set, model_name, verbose=1, results_dir="/work/results"):

    # TODO: Reduce the tolerance in np.isclose -->
    # generalisation: 100 images: 1e-2; noise: 10000 images: 1e-4 
    predictions_dir = os.path.join(results_dir, data_set, 'predictions')

    # compiled_stub = f"{model_name.rsplit(sep='_', maxsplit=1)[0]}_Set.csv"
    model_type, model_run = model_name.rsplit(sep='_', maxsplit=1)
    model_run = int(model_run)
    if model_type.endswith("ImageNet"):
        weights = "imagenet"
    else:
        weights = "None"

    # Generalisation image predictions
    n_classes = len(classes)
    n_images_per_class = 10
    y_generalise = np.repeat(range(n_classes), n_images_per_class)

    # Load consolidated accuracy records for checking
    # records_file = os.path.join(results_dir, "paper", f"generalise_{model_type}_Set.csv")
    records_file = os.path.join(results_dir, "paper", "generalise_Set.csv")
    df_records = pd.read_csv(records_file)
    df_records["Weights"].replace(np.nan, "None", inplace=True)

    frames = []
    for image_set in generalisation_sets:
        if image_set.endswith("inverted"):
            inverted = True
            image_set_type = image_set.rsplit(sep='_', maxsplit=1)[0]
        else:
            inverted = False
            image_set_type = image_set
        pred_file = os.path.join(predictions_dir, f"{model_name}_{image_set}.csv")
        probabilities = np.loadtxt(pred_file, delimiter=',', skiprows=0)
        classifications = np.argmax(probabilities, axis=1)
        assert len(classifications) == len(y_generalise)
        accuracy = sum(classifications == y_generalise) / len(y_generalise)
        mask = (df_records["Model"] == model_type) & \
                (df_records["Trial"] == model_run) & \
                (df_records["Weights"] == weights) & \
                (df_records["Set"] == image_set_type) & \
                (df_records["Inverted"] == inverted)

        if verbose > 1:
            print(f"{model_type=} {model_run=} {weights=} {image_set=} {inverted=}")
        recorded_accuracy = df_records[mask]["Accuracy"].values[0]
        if verbose:
            print(f"{image_set}: {accuracy:3.0%}")
        df_gen = pd.DataFrame(probabilities, columns=classes)
        df_gen["Predicted"] = classifications
        df_gen["Class"] = y_generalise
        df_gen["Correct"] = classifications == y_generalise
        df_gen["Image"] = range(n_classes * n_images_per_class)
        df_gen["Set"] = [image_set_type] * n_classes * n_images_per_class
        df_gen["Inverted"] = [inverted] * n_classes * n_images_per_class
        assert np.isclose(accuracy, recorded_accuracy), f"Calculated: {accuracy} =/= Recorded: {recorded_accuracy}"
        frames.append(df_gen)
    df_generalise = pd.concat(frames)

    # return df_generalise, df_records


    # Noise perturbation image predictions
    (_, _), (_, y_test) = cifar10.load_data()
    y_test = np.squeeze(y_test)

    # Check against:
    perturb_records_file = os.path.join(results_dir, "paper", f"perturb_{model_type}_Set.csv")
    df_perturb_records = pd.read_csv(perturb_records_file)
    df_perturb_records["Weights"].replace(np.nan, "None", inplace=True)

    n_levels = 11
    noise_types = get_perturbations(n_levels=n_levels)
    frames = []
    for noise, _, levels in noise_types:
        if verbose:
            print(f"{noise}: ", end="")
        for l_ind, level in enumerate(levels):
            pred_file = os.path.join(predictions_dir, f'{model_name}_{noise.replace(" ", "_").lower()}_L{l_ind+1:02d}.csv')
            probabilities = np.loadtxt(pred_file, delimiter=',', skiprows=0)
            classifications = np.argmax(probabilities, axis=1)
            assert len(classifications) == len(y_test)
            accuracy = sum(classifications == y_test) / len(y_test)

            mask = (df_perturb_records["Model"] == model_type) & \
                    (df_perturb_records["Trial"] == model_run) & \
                    (df_perturb_records["Weights"] == weights) & \
                    (df_perturb_records["Noise"] == noise) & \
                    (np.isclose(df_perturb_records["Level"], level))

            if verbose > 1:
                print(f"{model_type=} {model_run=} {weights=} {noise=} {level=}")
            recorded_accuracy = df_perturb_records[mask]["Accuracy"].values[0]
            # if verbose > 1:
            #     print(f"Calculated: {accuracy} =/= Recorded: {recorded_accuracy}")

            df_noise = pd.DataFrame(probabilities, columns=classes)
            df_noise["Predicted"] = classifications
            df_noise["Class"] = y_test
            df_noise["Correct"] = classifications == y_test
            df_noise["Image"] = range(len(y_test))
            df_noise["Noise"] = [noise] * len(y_test)
            df_noise["Level"] = [level] * len(y_test)
            df_noise["Level-Index"] = [l_ind] * len(y_test)
            assert np.isclose(accuracy, recorded_accuracy), f"Calculated: {accuracy} =/= Recorded: {recorded_accuracy}"
            # if not np.isclose(accuracy, recorded_accuracy, atol=1e-4, rtol=1e-5):
            #     print("TODO: Reinstate assertion!")
            #     print(f"Calculated: {accuracy} =/= Recorded: {recorded_accuracy}")
            frames.append(df_noise)

            if verbose:
                print(f"{accuracy:5.1%}", end=" ")
        if verbose:
            print()

    df_perturbation = pd.concat(frames)

    return df_generalise, df_perturbation


# def upscale_images(images, size, interpolation=None):
# """Images: (n_images, height, width, n_channels)"""
#     if interpolation is None:
#         x_train = x_train.repeat(7, axis=1).repeat(7, axis=2)
#         x_test = x_test.repeat(7, axis=1).repeat(7, axis=2)
#     else:
#         for image in images:
#             resized = cv2.resize(img, dim, interpolation=interpolation)

def calc_bandwidth(lambd, sigma):
    r = np.pi*sigma/lambd
    c = np.sqrt(np.log(2)/2)
    return np.log2((r + c)/(r - c))

def calc_sigma(lambd, bandwidth):
    p = 2**bandwidth
    c = np.sqrt(np.log(2)/2)
    return lambd * c / np.pi  * (p + 1) / (p - 1)

def calc_lambda(sigma, bandwidth):
    p = 2**bandwidth
    c = np.sqrt(np.log(2)/2)
    return sigma * np.pi / c  * (p - 1) / (p + 1)


def get_gabor_tensor(ksize, bs, sigmas, thetas, gammas, psis, lambdas=None, verbose=0):
    """Create a tensor of Gabor filters for greyscale images.
    
    The sinusoidal wavelength $\lambda$ is constrained by the bandwidth $b$
    and Gaussian spatial scale $\sigma$.
    """

    # n_kernels = len(sigmas) * len(thetas) * len(lambdas) * len(gammas) * len(psis)
    n_kernels = len(bs) * len(sigmas) * len(thetas) * len(gammas) * len(psis)
    gabors = []
    for sigma in sigmas:
        for theta in thetas:
            # for lambd in lambdas:
            for b in bs:
                lambd = calc_lambda(sigma, b)
                for gamma in gammas:
                    for psi in psis:
                        # params = {'ksize': ksize, 'sigma': sigma,
                        #           'theta': theta, 'lambd': lambd,
                        #           'gamma': gamma, 'psi': psi}
                        # gf = cv2.getGaborKernel(**params, ktype=cv2.CV_32F)
                        gf = cv2.getGaborKernel(ksize, sigma, theta, 
                                                lambd, gamma, psi, 
                                                ktype=cv2.CV_32F)
                        gf = K.expand_dims(gf, -1)
                        gabors.append(gf)
    assert len(gabors) == n_kernels
    if verbose:
        print(f"Created {n_kernels} kernels.")
        if verbose > 1:
            print("bs:", bs)
            print("sigmas:", sigmas)
            print("thetas:", thetas)
            print("gammas:", gammas)
            print("psis:", psis)
    return K.stack(gabors, axis=-1)  # (ksize[0], ksize[1], 1, n_kernels)


@tf.function
def convolve_tensor(x, kernel_tensor=None):
    '''
    conv2d
    input tensor: [batch, in_height, in_width, in_channels]
    kernel tensor: [filter_height, filter_width, in_channels, out_channels]
    '''
    # x = tf.image.rgb_to_grayscale(x)
    # print(f"Input shape: {x.shape}")
    # print(f"Kernel tensor shape: {kernel_tensor.shape}")  # TODO: Should this be (127, 127, 3, n)?

    # NOTE: This function does not apply bias or pass through an activation function
    # TODO: Do I need to add bias and pass through an activation function before returning? Maybe...
    return K.conv2d(x, kernel_tensor, padding='same')


def find_conv_layer(model, first=True):
    if first:
        layers = model.layers
    else:
        layers = reversed(model.layers)
    for ind, layer in enumerate(layers):
        if 'conv' in layer.name.lower():
        # if len(layer.output_shape) == 4:
            print(f"Found first convolutional layer {ind}: {layer.name}")
            layer_name = layer.name
            layer_ind = ind
            break
    return layer_name, layer_ind


def insert_noise_layer(model, layer=None, std=1, verbose=0):
    """Insert Gaussian Noise layer after the specified model layer."""

    assert 0 < std
    if layer is None:
        # Attempt to find the first convolutional layer
        _, layer = find_conv_layer(model)
    assert isinstance(layer, int)
    assert 0 < layer < len(model.layers)

    if verbose:
        print(f"Insert Gaussian noise layer with std={std} after layer {layer}...")

    for layer_ind, layer_object in enumerate(model.layers):
        # print(ind, layer.name)
        if layer_ind == 0:  # Get input layer
            config = layer_object.get_config()
            inp = Input(**config)
            # inp = layer
            x = inp
        else:
            x = tf.keras.layers.deserialize({'class_name': layer.__class__.__name__, 
                                             'config': layer.get_config()})(x)
        if layer_ind == layer:
            # Additive zero-centered Gaussian noise
            # As it is a regularization layer, it is only active at training time.
            x = tf.keras.layers.GaussianNoise(stddev=std)(x)

    model = Model(inputs=inp, outputs=x, name=f"noisy_{model.name}")
    return model


def substitute_layer(model, params, filter_type='gabor', replace_layer=1, 
                     input_shape=None, colour_input='rgb', 
                     use_initializer=False, noise_std=0, verbose=0):

    if replace_layer is None:
        # Attempt to find the first convolutional layer
        _, replace_layer = find_conv_layer(model)
    assert isinstance(replace_layer, int)
    assert 0 < replace_layer < len(model.layers)

    if filter_type.capitalize().startswith('Combined'):
        assert isinstance(params, dict)
        assert len(params) > 1
        configuration = params
        for layer_type in configuration:
            assert isinstance(configuration[layer_type], dict)
    else:
        # Assume an unnested dictionary has been passed
        configuration = {filter_type: params}

    for ind, layer in enumerate(model.layers):
        # print(ind, layer.name)
        if ind == 0:  # Get input layer
            config = layer.get_config()
            print(f"Replacing input: {config['batch_input_shape']} (batch) -->", end=' ')
            if colour_input == 'rgb':
                # inp = Input(shape=model.layers[0].input_shape[0][1:])
                if input_shape:
                    config['shape'] = input_shape + (3,)
                else:
                    config['shape'] = config['batch_input_shape'][1:]
            elif colour_input == 'rgba':
                print(f"Warning! colour_input: {colour_input} not yet implemented!")
                return
            elif colour_input == "grayscale":
                if input_shape:
                    config['shape'] = input_shape + (1,)
                else:
                    original_shape = config['batch_input_shape'][1:]
                    config['shape'] = (*original_shape[:-1], 1)
            else:
                raise ValueError(f"Unknown colour_input: {colour_input}")
            del config['batch_input_shape']
            inp = Input(**config)
            # inp = layer
            x = inp
            print(f"{config['shape']}")
        elif ind == replace_layer:
            for layer_type, params in configuration.items():
                if params is not None:  # Replace convolutional layer
                    print(f"Replacing layer {ind}: '{layer.name}' --> '{layer_type.lower()}_conv'...")
                    if verbose:
                        print(f"{layer_type.capitalize()} filter parameters:")
                        pprint(params)
                    if use_initializer:
                        if layer_type.lower() == 'gabor':
                            # Parse parameters
                            assert 'bs' in params
                            if 'sigmas' not in params:
                                assert 'lambdas' in params
                                # params['sigmas'] = [utils.calc_sigma(lambd, b) for lambd in params['lambdas']
                                #                     for b in params['bs']]
                            kernel_initializer = GaborInitializer(**params)
                        elif layer_type.lower() == 'dog':
                            kernel_initializer = DifferenceOfGaussiansInitializer(**params)
                        elif layer_type.lower() == 'low-pass':
                            kernel_initializer = LowPassInitializer(**params)
                        n_kernels = kernel_initializer.n_kernels
                        # When using this layer as the first layer in a model, provide the keyword argument 
                        # input_shape (tuple of integers, does not include the batch axis), 
                        # e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".

                        # Input shape: (batch, rows, cols, channels)
                        # Output shape: (batch, new_rows, new_cols, filters)
                        x = Conv2D(n_kernels, params['ksize'], padding='same', 
                                activation='relu', use_bias=True,
                                #    activation=None, use_bias=False,
                                name=f"{layer_type.lower()}_conv",
                                kernel_initializer=kernel_initializer, 
                                trainable=False)(x)
                    else:  # Deprecated
                        assert isinstance(layer, tf.keras.layers.Conv2D)
                        tensor = get_gabor_tensor(**params)  # Generate Gabor filters
                        x = Lambda(convolve_tensor, arguments={'kernel_tensor': tensor},
                                name=f"{layer_type.lower()}_conv")(x)
                else:
                    x = tf.keras.layers.deserialize({'class_name': layer.__class__.__name__, 
                                                    'config': layer.get_config()})(x)
                if noise_std:
                    # Add noise after the convolutional layer
                    x = tf.keras.layers.GaussianNoise(stddev=noise_std)(x)
        # elif ind == replace_layer + 1 and params is not None:  # Replace next layer
        #     # Check input_shape matches output_shape?
        #     # x = Conv2D(**layers[layer].get_config())(x)
        #     x = tf.keras.layers.deserialize({'class_name': layer.__class__.__name__, 
        #                                      'config': layer.get_config()})(x)
        else:
            # x = layer(x)
            x = tf.keras.layers.deserialize({'class_name': layer.__class__.__name__, 
                                             'config': layer.get_config()})(x)

    # del model
    if noise_std:
        new_name = f"{filter_type}_noise_{model.name}"
    else:
        new_name = f"{filter_type}_{model.name}"
    model = Model(inputs=inp, outputs=x, name=new_name)

    return model


def substitute_output(model, n_classes=16):

    if model.get_layer(index=-1).output_shape[-1] == n_classes:
        print(f"Model already has the requested number of output classes ({n_classes})!")
        return model

    new_model = []
    n_layers = len(model.layers)

    for ind, layer in enumerate(model.layers):
        if ind == 0:
            params = layer.get_config()
            params['shape'] = params['batch_input_shape'][1:]
            del params['batch_input_shape']
            inp = Input(**params)
            x = inp
        elif 0 < ind < n_layers-1:
            x = layer(x)
        else:  # Final layer
            params = layer.get_config()
            params['units'] = n_classes
            x = tf.keras.layers.deserialize({'class_name': layer.__class__.__name__, 
                                             'config': params})(x)

    return Model(inputs=inp, outputs=x, name=f"{model.name}_{n_classes}class")


# def add_top(model, n_fc_layers=3, n_classes=16):

#     new_model = []
#     for ind, layer in enumerate(model.layers):
#         new_model.append(layer)
    
#     for ind in range(n_fc_layers):
#         if ind < n_fc_layers-1:
#             name = f"fc{ind+1}"
#         else:
#             name = "predictions"
#         new_model.append(Dense())


class DifferenceOfGaussiansInitializer(Initializer):
    """Difference of Gaussians initializer class."""
    
#     def __init__(self, ksize, A_c, sigma_c, A_s=None, sigma_s=None, gamma=None):
#         if gamma is not None:
#             sigma_s = sigma_c * gamma
#         assert sigma_s > sigma_c

    # TODO: Remove ksize as it is redundant when passing shape to call()
    def __init__(self, ksize, sigmas, gammas, verbose=0):
        if isinstance(ksize, (int, float)):
            self.ksize = (ksize, ksize)
        else:
            assert len(ksize) == 2
            self.ksize = tuple(ksize)
        self.sigmas = sigmas
        self.n_kernels = len(sigmas) * len(gammas) * 2
        for gamma in gammas:
            assert gamma > 1
        self.gammas = gammas
        self.verbose = verbose

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. If not provided use the initializer
            dtype.
        """
        if self.verbose:
            print(f"Shape passed to DifferenceOfGaussiansInitializer: {shape=}")
        if shape is None:
            ksize = self.ksize
            n_channels_in = 1  # Assume monochrome inputs
            n_channels_out = self.n_kernels
        else:
            ksize = tuple(shape[:2])
            n_channels_in = shape[2]
            n_channels_out = shape[-1]
            assert self.ksize == ksize, f"[ksize] Passed: {ksize}; Expected: {self.ksize}"
            assert self.n_kernels == n_channels_out, f"[n_kernels] Passed: {n_channels_out}; Expected: {self.n_kernels}"

        kernels = []
        for sigma in self.sigmas:  # self.params['sigmas']:
            for gamma in self.gammas:
                sigma_c, sigma_s = sigma, sigma*gamma
                # Create centre
                kern_c_x = cv2.getGaussianKernel(ksize[0], sigma_c, ktype=cv2.CV_64F)
                kern_c_y = cv2.getGaussianKernel(ksize[1], sigma_c, ktype=cv2.CV_64F)
                kern_c = np.outer(kern_c_x, kern_c_y)
                kern_c /= np.sum(kern_c)

                # Create surround
                kern_s_x = cv2.getGaussianKernel(ksize[0], sigma_s, ktype=cv2.CV_64F)
                kern_s_y = cv2.getGaussianKernel(ksize[1], sigma_s, ktype=cv2.CV_64F)
                kern_s = np.outer(kern_s_x, kern_s_y)
                kern_s /= np.sum(kern_s)

                kern_on = kern_c - kern_s
                kern_off = kern_s - kern_c

                # kern_on = kern_on.astype('float32')  # HACK
                # kern_off = kern_off.astype('float32')  # HACK
                # kern_on = kern_on.astype(dtypes.as_dtype(dtype))
                # kern_off = kern_off.astype(dtypes.as_dtype(dtype))
                # kern_on = kern_on.astype(dtype)
                # kern_off = kern_off.astype(dtype)

                kern_on = K.expand_dims(kern_on, -1)
                kern_on = K.tile(kern_on, (1, 1, n_channels_in))
                kern_off = K.expand_dims(kern_off, -1)
                kern_off = K.tile(kern_off, (1, 1, n_channels_in))
                # kern_on = K.cast(kern_on, dtype)
                # kern_off = K.cast(kern_off, dtype)
                # kernels.append(kern)
                kernels.extend([kern_on, kern_off])
        # return K.stack(kernels, axis=-1)
        kernel_tensor = K.stack(kernels, axis=-1)

        if dtype:
            assert dtype in ('float16', 'float32', 'float64')
            if self.verbose:
                print(f"Casting to {dtype=}")
        else:
            dtype = 'float32'
        kernel_tensor = K.cast(kernel_tensor, dtype)

        return kernel_tensor

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.
        Returns:
        A JSON-serializable Python dict.
        """

        return {'ksize': self.ksize,
                'sigmas': self.sigmas,
                'gammas': self.gammas,
                }


class GaborInitializer(Initializer):
    """Gabor kernel initializer class."""

    def __init__(self, ksize, sigmas, bs, gammas, thetas, psis, verbose=0):
        # TODO: Deprecate ksize in (initialisation of) filter parameters
        if isinstance(ksize, (int, float)):
            self.ksize = (ksize, ksize)
        else:
            self.ksize = tuple(ksize)
#     def __init__(self, sigmas, bs, gammas, thetas, psis, verbose=0):
        self.sigmas = sigmas
        self.bs = bs
        self.gammas = gammas
        self.thetas = thetas
        self.psis = psis
        self.n_kernels = len(sigmas) * len(bs) * len(gammas) * len(thetas) * len(psis)
        self.verbose = verbose

    # TODO: Use @property decorator with setter and getter methods
    # def calc_n_kernels(params):
    #     return len(params['bs']) * len(params['sigmas']) * len(params['thetas']) \
    #                              * len(params['gammas']) * len(params['psis'])

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. If not provided use the initializer
            dtype.
            partition_info: Optional information about the possible partitioning of a
            tensor.
        """
        # return get_gabor_tensor(ksize, bs, sigmas, thetas, gammas, psis, lambdas=None)
        # return get_gabor_tensor(**self.params)
        if self.verbose:
            print(f"Shape passed to GaborInitializer: {shape=}")
        if shape is None:
            ksize = self.ksize
            n_channels_in = 1  # Assume monochrome inputs
            n_channels_out = self.n_kernels
        else:
            ksize = tuple(shape[:2])
            n_channels_in = shape[2]
            n_channels_out = shape[-1]
            assert self.ksize == ksize, f"[ksize] Passed: {ksize}; Expected: {self.ksize}"
            assert self.n_kernels == n_channels_out, f"[n_kernels] Passed: {n_channels_out}; Expected: {self.n_kernels}"
        # print(f'{ksize=}')
        # return get_gabor_tensor(ksize, self.bs, self.sigmas, 
        #                         self.thetas, self.gammas, self.psis, self.verbose)

        gabors = []
        for sigma in self.sigmas:
            for theta in self.thetas:
                # for lambd in lambdas:
                for b in self.bs:
                    lambd = calc_lambda(sigma, b)
                    for gamma in self.gammas:
                        for psi in self.psis:
                            gf = cv2.getGaborKernel(ksize, sigma, theta, 
                                                    lambd, gamma, psi, 
                                                    # ktype=cv2.CV_32F)
                                                    ktype=cv2.CV_64F)
                            gf = K.expand_dims(gf, -1)
                            gf = K.tile(gf, (1, 1, n_channels_in))  # Generalise for multi-channel inputs
                            gabors.append(gf)
        assert len(gabors) == self.n_kernels
        if self.verbose:
            print(f"Created {self.n_kernels} kernels.")
            if self.verbose > 1:
                print("bs:", self.bs)
                print("sigmas:", self.sigmas)
                print("thetas:", self.thetas)
                print("gammas:", self.gammas)
                print("psis:", self.psis)

        # print(gf.get_shape())
        gf_tensor = K.stack(gabors, axis=-1)  # (ksize[0], ksize[1], 1, n_kernels)
        if dtype:
            assert dtype in ('float16', 'float32', 'float64')
            if self.verbose:
                print(f"Casting to {dtype=}")
        else:
            dtype = 'float32'
        gf_tensor = K.cast(gf_tensor, dtype)  # tf.keras.backend.cast
        if self.verbose:
            print(f"Generated tensor shape: {gf_tensor.get_shape()}")
        return gf_tensor

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.
        Returns:
        A JSON-serializable Python dict.
        """
        # return {"dtype": self.dtype.name ...}
        # return self.params
        return {'ksize': self.ksize,
                'sigmas': self.sigmas,
                'bs': self.bs,
                'gammas': self.gammas,
                'thetas': self.thetas,
                'psis': self.psis
                }


# from preparation import low_pass_filter

class LowPassInitializer(Initializer):
    """Low Pass filter Initializer class."""

    def __init__(self, ksize, sigmas, verbose=0):  #params, dtype=dtypes.float32):
        # self.dtype = dtypes.as_dtype(dtype)
        # self.params = params
        # TODO: Deprecate ksize in (initialisation of) filter parameters
        if isinstance(ksize, (int, float)):
            self.ksize = (ksize, ksize)
        else:
            self.ksize = tuple(ksize)
        self.sigmas = sigmas
        self.n_kernels = len(self.sigmas)
        self.verbose = verbose

    def __call__(self, shape, dtype=None):  #, partition_info=None):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. If not provided use the initializer
            dtype.
            partition_info: Optional information about the possible partitioning of a
            tensor.
        """

        # if dtype is None:
        #     dtype = 'float32'  # floatx()
        # ksize = self.params['ksize']
        if shape is None:
            ksize = self.ksize
            n_channels_in = 1  # Assume monochrome inputs
            n_channels_out = self.n_kernels
        else:
            ksize = tuple(shape[:2])
            n_channels_in = shape[2]
            n_channels_out = shape[-1]
            assert self.ksize == ksize, f"[ksize] Passed: {ksize}; Expected: {self.ksize}"
            assert self.n_kernels == n_channels_out, f"[n_kernels] Passed: {n_channels_out}; Expected: {self.n_kernels}"
        # assert ksize == shape[0] == shape[1]

        # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
        # Method 1
        # # create nxn zeros
        # inp = np.zeros((kernel_size, kernel_size))
        # # set element at the middle to one, a dirac delta
        # inp[kernel_size//2, kernel_size//2] = 1
        # # gaussian-smooth the dirac, resulting in a gaussian filter mask
        # return low_pass_filter(inp, std)[:,:,0]

        # Method 2
        kernels = []
        for sigma in self.sigmas:  # self.params['sigmas']:
            kern_x = cv2.getGaussianKernel(ksize[0], sigma)
            kern_y = cv2.getGaussianKernel(ksize[1], sigma)
            kern = np.outer(kern_x, kern_y)
            kern /= np.sum(kern)
            # kern = kern.astype('float32')  # HACK
            # kern = kern.astype(dtypes.as_dtype(dtype))
            # kern = kern.astype(dtype)
            kern = K.expand_dims(kern, -1)
            kern = K.tile(kern, (1, 1, n_channels_in))  # Generalise for multi-channel inputs
            kernels.append(kern)
        # return K.stack(kernels, axis=-1)
        tensor = K.stack(kernels, axis=-1)
        if dtype:
            assert dtype in ('float16', 'float32', 'float64')
            if self.verbose:
                print(f"Casting to {dtype=}")
        else:
            dtype = 'float32'
        tensor = K.cast(tensor, dtype)
        return tensor

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.
        Returns:
        A JSON-serializable Python dict.
        """
        # return {"dtype": self.dtype.name ...}
        # return self.params
        return {'ksize': self.ksize,
                'sigmas': self.sigmas}


class KernelInitializer(Initializer):
    """Kernel initializer class for Conv2D layers."""

    def __init__(self, params, kernel_function, dtype=dtypes.float32):
        self.dtype = dtypes.as_dtype(dtype)
        self.params = params
        self.function = kernel_function
        # self.n_kernels = self.calc_n_kernels(params)
    
    def calc_n_kernels(self, params):
        return np.prod([value for param, value in params.items() if param != 'ksize'], dtype=int)

    def __call__(self, shape, dtype=None, partition_info=None):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. If not provided use the initializer
            dtype.
            partition_info: Optional information about the possible partitioning of a
            tensor.
        """
        return self.function(**self.params)

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.
        Returns:
        A JSON-serializable Python dict.
        """
        # return {"dtype": self.dtype.name ...}
        return self.params


def load_model(data_set, name, project_root_dir=None, verbose=0):
    # TODO: Check shape and dtype work
    # TODO: Restore optimizer state (and ReduceLR)
#     if "project_root_dir" in globals():  # Set in globals
#         project_root_dir = globals()["project_root_dir"]
    if project_root_dir:
        models_dir = os.path.join(project_root_dir, "models")
    else:
        models_dir = "~/work/BioNet/models"
#     path_to_model = f"/work/models/{data_set}/{name}"
    path_to_model = os.path.join(models_dir, data_set, name)

    if verbose:
        print(f"Loading model: {path_to_model}", flush=True)
    sim = get_simulation_params(data_set, name, models_dir)
    
    stub = f"{sim['epochs']}_epochs"
    
    # Retrofit new variable names
    if 'convolution' in sim:
        convolution = sim['convolution']
    elif 'conv' in sim:
        convolution = sim['conv']
    else:
        if sim['model'].capitalize().startswith('Gabor'):
            convolution = 'Gabor'
        elif sim['model'].capitalize().startswith('Low-pass'):
            convolution = 'Low-pass'
        else:
            convolution = 'Original'
    if verbose > 1:
        pprint(sim)
    filter_params = sim['filter_params']

    if convolution == 'Gabor':
        custom_objects = {'GaborInitializer': GaborInitializer(**filter_params)}
    elif convolution == 'DoG':
        custom_objects = {'DifferenceOfGaussiansInitializer': DifferenceOfGaussiansInitializer(**filter_params)}
    elif convolution == 'Low-pass':
        custom_objects = {'LowPassInitializer': LowPassInitializer(**filter_params)}
    elif convolution.startswith('Combined'):
        custom_objects = {'DifferenceOfGaussiansInitializer': DifferenceOfGaussiansInitializer(**filter_params['DoG']),
                          'GaborInitializer': GaborInitializer(**filter_params['Gabor'])}
    else:
        custom_objects = None
    if verbose:
        print("Parameters: ")
        pprint(filter_params)
        
    with open(os.path.join(path_to_model, f"{stub}.json"), "r") as sf:
        config = sf.read()
    if verbose > 1:
        print("Model config: ")
        pprint(config)
        print("Custom objects: ")
        pprint(custom_objects)
    # with CustomObjectScope(custom_objects):
    model = tf.keras.models.model_from_json(config, custom_objects)
    model.load_weights(os.path.join(path_to_model, f"{stub}_weights.h5"))
    if verbose:
        model.summary()

    return model



# def consolidate_results(pattern, output):
# #     pattern = patterns['DoG']['ALL-CNN']
# #     output = f'/work/results/paper/perturb_DoG_ALL-CNN_Set.csv'
#     df = pd.concat([pd.read_csv(file) for file in glob.glob(pattern)], ignore_index=True)
#     df["Weights"] = "None"
#     df["Model"].where(df == "VGG16") = "VGG-16"
#     df["Model"].where(df == "VGG19") = "VGG-19"
# #     df.mask("VGG16" in df, "VGG-16") = "VGG-16"
#     if "ImageNet" in pattern:
#         df["Weights"] = "imagenet"
#     else:
#         df["Weights"] = "None"
#     df["Convolution"] = "Original"
#     df["Base"] = "VGG19"
#     df["Model"] = "VGG19_ImageNet"
#     df = df[columns]
#     df.sort_values(by=['Trial', 'Noise', 'Level'], inplace=True, ignore_index=True)
#     df.to_csv(output, index=False)
# #     frames.append(pd.read_csv(output))
#     return pd.read_csv(output)


def get_perturbation_results(tag):
    # TODO: Overhaul
    frames = []
    columns = ['Trial', 'Model', 'Convolution', 'Base', 'Weights', 
               'Noise', 'Level', 'Loss', 'Accuracy']

    # Gaussian Noise
    tag = 'noise_s_0_2'
    sigma = 0.2
    seed =  1895185933 #1086513891  #2817631224
    pattern = f'/work/results/{tag}/perturb_*s{seed}.csv'
    output = f'/work/results/{tag}/perturb_noise_{sigma}_Set.csv'
    df = pd.concat([pd.read_csv(file) for file in glob.glob(pattern)], ignore_index=True)
    df["Weights"] = "None"
    df = df[columns]
    df.sort_values(by=['Trial', 'Noise', 'Level'], inplace=True, ignore_index=True)
    df.to_csv(output, index=False)
    frames.append(pd.read_csv(output))

    perturbations = pd.concat(frames, ignore_index=True)

    return perturbations


def calc_aucs(df, verbose=1):
    
    # noise_types = df.Noise.unique().tolist()
    noise_types = ["Uniform", "Salt and Pepper", "High Pass", "Low Pass", 
                   "Contrast", "Phase Scrambling", "Darken", "Brighten", 
                   "Rotation", "Invert"]
    models = df.Model.unique().tolist()
    convolutions = df.Convolution.unique().tolist()
    bases = df.Base.unique().tolist()

    if verbose > 1:
        print(noise_types)
        print(models)
        print(convolutions)
        print(bases)

    scores = {}
    for noise in noise_types:
        scores[noise] = {}
        for model in models:
            query = f"Noise == '{noise}' and Model == '{model}'"
            x = df.query(f"{query} and Trial == 1").Level.to_numpy()
            y = df.query(query).groupby('Level').mean().Accuracy.to_numpy()  # Average over trials

            scores[noise][model] = auc(x, y)
            if verbose:
                print(f'{noise:16} | {model:14}: AUC = {scores[noise][model]:6.3f} | SIMP = {simps(y, x):7.3f}')
        if verbose:
            print('-' * 64)
    return scores
