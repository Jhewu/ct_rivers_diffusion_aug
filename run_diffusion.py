""" 
THIS DDIM MODEL TRAINS RIVER IMAGES
200X600X3 IMAGES, ONE MODEL AT A TIME, 
IF YOU WANT TO TRAIN MULTIPLE MODELS, 
PLEASE USE RUN_TRAINING_LOOP.PY

THE CURRENT VERSIONS USED ARE: 
TENSORFLOW 2.16.1 OR 2.17
AND KERAS 3.6

PLEASE REFER TO THE
OFFICIAL KERAS WEBSITE
https://keras.io/getting_started/
"""

# Import necessary libraries
import os
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
from keras import mixed_precision
import cv2 as cv
import datetime
import shutil

# Import from local script
from diffusion_model import DiffusionModel
from config import Config
from utils import *

def PrepareData_LoadModel(config):
    # Load and prepare the dataset
    train_dataset, val_dataset = load_dataset(config.in_dir, config.validation_split, 
                                              config.seed, config.image_size)
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset, config.batch_size)
    
    # Create and compile the model
    model = DiffusionModel(config.image_size, config.widths, config.block_depth, 
                           config.eta, config.max_signal_rate, config.min_signal_rate, 
                           config.batch_size, config.ema, config.embedding_dims,
                           config.attention_in_up_down_sample, config.attention_in_bottleneck)

    # Calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(train_dataset)

    return train_dataset, val_dataset, model

def TrainDiffusionModel(config, model, train_dataset, val_dataset, 
                        generate_image_callback, last_checkpoint_callback, 
                        checkpoint_callback, early_stop_callback, csv_callback):

    # Implement mixed_precision if dictated
    if config.use_mix_precision: 
        mixed_precision.set_global_policy("mixed_float16")
        print("\nUsing mixed_precision float16\n")
    else: 
        mixed_precision.set_global_policy("float32")
        print("\nDefault, using float32\n")

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=config.learning_rate, weight_decay=config.weigth_decay
        ),
        # Loss function: Pixelwise mean absolute error (MAE).
        loss=keras.losses.mean_absolute_error,
    )

    # Load the model if desired
    if config.load_and_train: 
        model.load_weights(f"{config.model_dir}/best_diffusion_model.weights.h5")
        config.out_dir = config.model_dir

    # Create output directory if it does not exist
    CreateDir(config.out_dir)

    # Copy parameters file into the model folder
    shutil.copy("config.py", f"{config.out_dir}/")
    print(f"\nConfig copied to {config.out_dir}/")

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=config.num_epochs,
        validation_data=val_dataset,
        callbacks=[
            early_stop_callback,
            csv_callback,
            generate_image_callback,
            checkpoint_callback, 
            last_checkpoint_callback, 
        ],
    )

    # Save the key loss metrics 
    save_history(history, config.out_dir)
    
def InferenceDiffusionModel(model): 
    # Load the model's weights
    model.load_weights(f"{config.model_dir}/best_diffusion_model.weights.h5")

    # Generate the images
    generated_images = model.generate(config.images_to_generate, config.generate_diffusion_steps)

    # Create directory in model's folder and save the images
    if config.subprocess: 
        # When running diffusion_augmentation.py
        # we want to save the images to the specified out_dir
        # not in the model's folder
        generated_dir = config.out_dir

    else:
        generated_dir = os.path.join(config.model_dir, "generated_images")

    if not os.path.exists(generated_dir): 
        os.makedirs(generated_dir)

    # Get today's date
    current_date = datetime.datetime.now() 
    formatted_date = current_date.strftime("%m%d%y")

    # Get label
    label = os.path.basename(config.in_dir)

    # Save the images in the folder with the 
    # corresponding naming convention
    index = 1
    for image in generated_images: 
        image_name = f"S0_D{formatted_date}_{formatted_date}_0_{index}_DM_AUG_{label}.JPG"
        tf.keras.preprocessing.image.save_img(f"{generated_dir}/{image_name}", image) 
        index = index + 1

def ContextualInpainting(model, config): 
    """
    This function is used for Contextual inpainting with a pre-trained
    diffusion model. The specific method implemented is from the paper: 
    "RePaint: Inpainting using Denoising Diffusion Probabilistic Models"
    https://arxiv.org/abs/2201.09865 

    Requires: 
    1. Pre-trained model in parameters.py
    2. Requires directories mask_and_image/images and mask_and_image/masks. 
       Each mask and image pair, must contain the same name
    """

    """
    THIS CODE NEEDS MODIFICATION. RIGHT NOW IT ONLY TAKES ONE IMAGE
    IN THE FUTURE MODIFY TO TAKE ENTIRE DIRECTORIES. DO NOT USE AS OF 
    THE MOMENT
    """
    # Load the model's weights
    model.load_weights(f"{config.model_dir}/best_diffusion_model.weights.h5")

    # Load masks and images
    """MODIFY TO HANDLE LONG LISTS OF IMAGES AND MASKS"""
    image_list, mask_list, image_dir, mask_dir = load_inpainting_data_temp(config.inpainting_dir)
    image_name = image_list[0]
    mask_name = mask_list[0]

    image_path = os.path.join(image_dir, image_name)
    mask_path = os.path.join(mask_dir, mask_name)

    image = cv.imread(image_path, cv.IMREAD_COLOR)
    mask = cv.imread(mask_path, cv.IMREAD_COLOR)

    # Run RePaint algorithm
    inpainted_images = model.repaint(image, mask, diffusion_steps=50)

    plt.imshow(inpainted_images[0])
    plt.show()

    """MODIFY TO HANDLE LONG LISTS OF IMAGES AND MASKS"""
    # Save the inpainted image in model directory
    inpainted_dir = os.path.join(config.out_dir, "inpainted_images")
    if not os.path.exists(inpainted_dir): 
        os.makedirs(inpainted_dir)
    """MODIFY THE NAMING CONVENTION TO FIT INTO THE PIPELINE""" 

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    for image in inpainted_images: 
        tf.keras.preprocessing.image.save_img(f"{inpainted_dir}/{timestamp}_inpainted_img_{image_name}.jpg", image) 

if __name__ == "__main__":
    # Warn the user of the version compatibilities
    print(keras.__version__)
    print(tf.__version__)
    print("\nWARNING: If Keras version is not 3.6.0, and Tensorflow 2.16.1 or 2.17.0 you might run into issues\n")

    # Set backend to tensorflow
    os.environ["KERAS_BACKEND"] = "tensorflow"

    # Ensure that the model does not take all the GPU memory
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\nThese are the current GPUs detected in the system:\n {gpus}\n")

    # Obtain the config file
    config = Config()

    if gpus: 
        try: 
            # Choose the GPU to use
            tf.config.set_visible_devices(gpus[config.gpu_index], "GPU")
            print(f"Currently using {gpus[config.gpu_index]}\n")

            # Set memory growth
            tf.config.experimental.set_memory_growth(gpus[config.gpu_index], True)
            print(f"Set to growing memory\n")
        except RuntimeError as e: print(e)

    # Only error messages will be displayed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    des=""" 
    ------------------------------------------
    - CT Rivers Diffusion Models (Overview) - 

    This project aims to improve image classification
    accuracy for CT river images. By using diffusion model
    to artificially augment or dataset, we can ensure a 
    balanced dataset, often not the case in environmental images, 
    and expose the classification model to more "diversity."

    ------------------------------------------
    - Changing Parameters for the Diffusion Model -

    There are two ways to change parameters: 

    (1) Directly modify the config.py
    (2) Use argparse to pass arguments. NOTE: not all parameters are
        available for argparse, if you wish to modify them, modify the 
        config.py directly. 

    ------------------------------------------
    - How to Use -

    Please refer to the README.md file for the full list of argparse argument. 
    If you're working with GUI, I highly recommend changing the parameters using
    the config.py by manually modifying the values.

    To train the model (using config.py): 
    (1) Set runtime = "training" 
    (2) Set in_dir to the dataset directory + the label, e.g. diffusion_data/L2
    (3) Modify all other important training parameters, such as
        num_epochs, batch_size or learning_rate
    (3) (in command prompt) 'python3 run_diffusion.py'
    
    To train the model (using argparse)
    (1) (default parameters) 'python3 run_diffusion.py"
    (2) (different parameters) 'python3 run_diffusion.py --in_dir diffusion_data/L2
        --learning_rate 1e-5 --batch_size 8

    ---

    To perform inference (using config.py): 
    (1) You need to have a trained a model
    (2) Modify model_dir to the folder where the weight is located e.g. 
        "results/L2_2025-01-12_14:55:51"
    (3) Modify all other important parameters such as images_to_generate
        and generate_diffusion_steps 

    To perform inference (using argparse): 
    (1) (default parameters) Need to provide the model_dir: 
        'python3 run_diffusion --runtime inference
        --model_dir results/L2_2025-01-12_14:55:51'
    (2) (changing parameters): 
        'python3 run_diffusion --runtime inference
        --in_dir diffusion_data/L2
        --model_dir results/L2_2025-01-12_14:55:51'

    NOTE: to perform inference you still need to provide the correct
    in_dir, because the model need to adapt the dataset for the normalizer
    to work correctly. The architecture also needs to be the same otherwise
    the weights cannot be loaded. Since the image generation follows a convention, 
    if you generate 30, and generate 30 again, the files will be replaced by the 
    latest. The differentiator is today's date. 

    Please refer to the README.md file for more information on how to use
    or access help, by 'python3 run_diffusion.py -h'

    ------------------------------------------
    """

    # Parse the arguments
    args = ParseArgs(des) 

    # Assign arguments
    config = AssignArgs(config, args)

    # Load the previous architectural configuration without altering
    # the current hyperparameters
    config = LoadPrevConfig(config, f"{config.model_dir}/config_file.pkl")

    if config.runtime == "training":
        # Get current time to create output directory
        current_time = datetime.datetime.now() 
        formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
        basename = os.path.basename(config.in_dir)
        config.out_dir = os.path.join(config.out_dir, f"{basename}_{formatted_time}")

        # Prepare Dataset and Load Model
        train_dataset, val_dataset, model = PrepareData_LoadModel(config)

        # Get callbacks
        generate_image_callback, last_checkpoint_callback, checkpoint_callback, early_stop_callback, csv_callback = GetCallbacks(config)

        # Train the Diffusion Model
        TrainDiffusionModel(config, model, train_dataset, val_dataset, 
                        generate_image_callback, last_checkpoint_callback, 
                        checkpoint_callback, early_stop_callback, csv_callback)
               
        # Perform Inference after training
        if not config.use_mix_precision: 
            mixed_precision.set_global_policy("mixed_float16")
            print("Using mixed_precision float16\n")
        else: 
            mixed_precision.set_global_policy("float32")
            print("\nDefault, using float32\n")
        InferenceDiffusionModel(model)        

        # Log the parameters of config and save as a pickle file
        log_config_parameters(config, f"{config.model_dir}/config_parameters.txt")
        save_config_to_pickle(config, f"{config.model_dir}/config_file.pkl")

        print(f"\nFinish training for {config.num_epochs}. Check {config.out_dir} for the model")
                            
    elif config.runtime == "inference":
        if not os.path.exists(config.out_dir): 
            raise Exception("\nWARNING: Cannot find the directory where the model and all its files are stored\n")

        # Prepare Dataset and Load Model
        _, _, model = PrepareData_LoadModel(config)
        
        # Perform inference
        InferenceDiffusionModel(model)        
        print(f"\n Finish generating {config.images_to_generate} images. Check {config.model_dir} for the images\n")

    elif config.runtime == "inpainting":
        if not os.path.exists(config.out_dir): 
            raise Exception("\nWARNING: Cannot find the directory where the model and all its files are stored\n")
        
        print("\nInpainting is currently not fully implemented yet, please refrain from using it\n")

        # Prepare Dataset and Load Model
        _, _, model = PrepareData_LoadModel(config)
                
        # ContextualInpainting(model, config)
        print(f"\nFinish inpainting images\n")
    
    # Clear all used GPU memory
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph() 
