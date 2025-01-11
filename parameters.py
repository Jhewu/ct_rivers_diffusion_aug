"""
THIS PYTHON FILE ONLY CONTAINS THE PARAMETERS
FOR THE DIFFUSION MODELS. THIS IS DONE TO AVOID
CIRCULAR IMPORT. 
"""

import os

""" GENERAL PARAMETERS """
folder_name = "ct_deep_river_diffusion"
img_folder_name = "flow_large"            # must be in cwd
                                    # where the training dataset is
folder_path = "all_exp/exp5"
label = "L2"

run_description = "Testing if the modification works"

# Create the folder if it exists
if not os.path.exists(folder_path): 
    os.makedirs(folder_path)

""" TRAINING PARAMETERS """
# TRAINING PARAMETERS
runtime = "training"
                                    # if it's "training," it's in training mode
                                    # if it's "inference," it's in inference mode
                                    # if It's "inpaint," it's in inpainting mode
load_and_train = False
eta = 0.5

# MODEL PARAMETERS
image_size = (200,600)
                                     # for 200x600 is 200, 600 (for 2 downsampling blocks)

# preprocessing
seed = 42
validation_split = 0.15
pad_to_aspect_ratio = False
crop_to_aspect_ratio = True     

# optimization
num_epochs = 1
batch_size = 4
dataset_repetitions = 1
ema = 0.999
learning_rate = 2.5e-4
weight_decay = learning_rate/10
used_mix_precision = True

# sampling
min_signal_rate = 0.01
max_signal_rate = 0.95

# u-net architecture
embedding_dims = 128
widths = [32, 64, 96, 128]
block_depth = 2
attention_in_bottleneck = False
attention_in_up_down_sample = False

# callback param
checkpoint_monitor = "n_loss"
early_stop_monitor = "n_loss"
early_stop_min_delta = learning_rate/10
early_stop_patience = 25
early_stop_start_epoch = 50
generate_on_epoch = 100000

""" INFERENCE PARAMETERS """
images_to_generate = 5
generate_diffusion_steps = 30

""" INPAINTING PARAMETERS """
inpainting_dir = "inpainting_data"
MASK_AND_IMAGE_DIR = "mask_and_image"








