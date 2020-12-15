import os

# Paths to data, saved logs and weights.
DATA_PATH = 'data'
LOGS_DIR = 'logs'
WEIGHTS_PATH = 'weights'
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train_corr.npy')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_corr.npy')

# Training parameters.
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 100

# Custom model parameters.
ALPHA = 1.0
REGULARIZATION = 0.0005
ACTIVATION_TYPE = 'leaky'

MODEL_TYPE = 'custom_resnet18'
CLASS_NAMES = ('True signal', 'False signal')
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SHAPE = (32, 32, 1)
INPUT_NAME = 'input'
OUTPUT_NAME = 'output'
