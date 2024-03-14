import os
import sys
import pprint
import subprocess
import random
from tqdm.notebook import tqdm
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend
project_root_dir = "/home/asd/Downloads/BioNet/"  # Change as necessary
print(f"Project directory: {project_root_dir}\n")
sys.path.append(project_root_dir)
print("\nTensorFlow:", tf.__version__)
print(f"Channel ordering: {tf.keras.backend.image_data_format()}")  # TensorFlow: Channels last order.
gpus = tf.config.experimental.list_physical_devices('GPU')
# gpus = tf.config.list_physical_devices('GPU')
pprint.pprint(gpus)
label = "paper"
image_path = ''  # Empty string defaults to CIFAR-10
# image_path = '/shared/data/ecoset-cifar10'
convolutions = ['Original', 'Low-pass', 'DoG', 'Gabor', 'Combined-trim']
bases = ['ALL-CNN', 'VGG-16', 'VGG-19', 'ResNet']
seed = 0
start_trial = 1
num_trials = 5
trials = range(start_trial, start_trial+num_trials)
train = True
pretrain = False
clean = False
epochs = 100
optimizer = "RMSprop"
lr = 1e-4
use_initializer = True
data_augmentation = True
extra_augmentation = False
internal_noise = 0
skip_test = False
save_images = False
save_predictions = True
test_generalisation = True
test_perturbations = True
interpolation = 4  # Lanczos
recalculate_statistics = False
verbose = 0
halt_on_error = False
gpu = 0
######################################
script = os.path.join(project_root_dir, "model.py")
flags = ['--log']
if train:
    flags.append('-t')
if clean:
    flags.append('-c')
if use_initializer:
    flags.append('--use_initializer')
if data_augmentation:
    flags.append('--data_augmentation')
if extra_augmentation:
    flags.append('--extra_augmentation')
if skip_test:
    flags.append('--skip_test')
if recalculate_statistics:
    flags.append('--recalculate_statistics')
if save_predictions:
    flags.append('--save_predictions')
optional_args = []
if image_path:
    optional_args.extend(['--image_path', str(image_path)])
if test_perturbations:
    optional_args.append('--test_perturbations')
if test_generalisation:
    optional_args.append('--test_generalisation')
if pretrain:
    optional_args.append('--pretrain')
if internal_noise:
    optional_args.extend(['--internal_noise', str(internal_noise)])
if interpolation:
    optional_args.extend(['--interpolation', str(interpolation)])
if verbose:
    optional_args.extend(['--verbose', str(verbose)])
count = 1
for trial in tqdm(trials, desc='Trial'):
    if seed is None:
        seed = random.randrange(2**32)
    for base in tqdm(bases, desc='Model Base', leave=False):
        for conv in tqdm(convolutions, desc='Convolution', leave=False):
            cmd = [script, *flags]
            if save_images and count == 1:
                cmd.append('--save_images')
            cmd.extend(['--convolution', conv, '--base', base, '--label', label,
                        '--trial', str(trial), '--seed', str(seed),
                        '--optimizer', optimizer, '--lr', str(lr),
                        '--epochs', str(epochs), '--gpu', str(gpu)])
            cmd.extend(optional_args)
            completed = subprocess.run(cmd, shell=False, capture_output=True, text=True)
            if completed.returncode != 0:
                print(completed.stdout)
                print(completed.stderr)
            count += 1
f'Finished job "{label}"!'