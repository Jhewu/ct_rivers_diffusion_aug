"""
THIS PYTHON FILE ONLY CONTAINS THE PARAMETERS
FOR THE DIFFUSION MODELS. THIS IS DONE TO AVOID
CIRCULAR IMPORT. 
"""

import logging
import os

""" GENERAL PARAMETERS """
folder_name = "ct_deep_river_diffusion"
img_folder_name = "flow_large"            # must be in cwd
                                    # where the training dataset is
folder_path = "all_exp/exp4"
label = "L2"

run_description = "Added multihead attention to the bottleneck, and added single head attention into the downsampling and upsampling blocks"

# Create the folder if it exists
if not os.path.exists(folder_path): 
    os.makedirs(folder_path)

# Configure logging
logging.basicConfig(filename=f'{folder_path}/model_parameters.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
validation_split = 0.2
pad_to_aspect_ratio = False
crop_to_aspect_ratio = True     

# optimization
num_epochs = 100
batch_size = 8
dataset_repetitions = 1
ema = 0.999
learning_rate = 3.5e-4
weight_decay = learning_rate/10

# KID eval
kid_image_size = 75
plot_diffusion_steps = 20
kid_diffusion_steps = 5

# sampling
min_signal_rate = 0.01
max_signal_rate = 0.95

# u-net architecture
embedding_dims = 256
widths = [64, 96, 128, 256]
block_depth = 3

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

# Log all variables
logging.info("----------")
logging.info(f"RUN DESCRIPTION:\n {run_description}\n")
logging.info(f'GENERAL PARAMETERS:\nfolder_name: {folder_name}\nimg_folder_name: {img_folder_name}\nfolder_path: {folder_path}')
logging.info("")
logging.info(f'TRAINING PARAMETERS:\nmode: {runtime}\nload_and_train: {load_and_train}')
logging.info("")
logging.info(f'MODEL PARAMETERS:\nimage_size: {image_size}\nseed: {seed}\nvalidation_split: {validation_split}\npad_to_aspect_ratio: {pad_to_aspect_ratio}\ncrop_to_aspect_ratio: {crop_to_aspect_ratio}\nnum_epochs: {num_epochs}\nbatch_size: {batch_size}\ndataset_repetitions: {dataset_repetitions}\nema: {ema}\nlearning_rate: {learning_rate}\nweight_decay: {weight_decay}\nkid_image_size: {kid_image_size}\nplot_diffusion_steps: {plot_diffusion_steps}\nkid_diffusion_steps: {kid_diffusion_steps}\nmin_signal_rate: {min_signal_rate}\nmax_signal_rate: {max_signal_rate}\nembedding_dims: {embedding_dims}\nwidths: {widths}\nblock_depth: {block_depth}')
logging.info("")
logging.info(f'CALLBACK PARAMETERS:\ncheckpoint_monitor: {checkpoint_monitor}\nearly_stop_monitor: {early_stop_monitor}\nearly_stop_min_delta: {early_stop_min_delta}\nearly_stop_patience: {early_stop_patience}\nearly_stop_start_epoch: {early_stop_start_epoch}')
logging.info("")
logging.info(f'INFERENCE PARAMETERS:\nimages_to_generate: {images_to_generate}\ngenerate_diffusion_steps: {generate_diffusion_steps}')
logging.info("----------")








