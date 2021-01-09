# Dataset info

IMG_SIZE = 256
CHANNELS = 3

# Hyperparams
LEARNING_RATE = 2e-4

# Training

# Weights for loss functions
CYCLE_WEIGHT = 10
IDT_WEIGHT = 0.5 * CYCLE_WEIGHT
ADV_WEIGHT = 1

BATCH_SIZE = 1 # Batch size for training
USE_CUDA = True
SAMPLE_INTERVAL = 100 # Interval at which to draw samples
CHECKPOINT_INTERVAL = 500 # Interval at which to save checkpoints
LOG_INTERVAL = 50 # Interval at which to print log
LOAD_CHECKPOINTS = False
HALF_PRECISION = False # 16 bit floats

SCHEDULER_OFFSET = 0
