""" 
THIS DDIM MODEL TRAINS RIVER IMAGES
200X600X3 IMAGES

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
from datetime import datetime
import shutil

# Import from local script
from diffusion_model import DiffusionModel
from callbacks import * 
from parameters import *
from utils import *

"""-----MAIN RUNTIME BELOW-----"""
def TrainDiffusionModel():
    # Load and prepare the dataset
    train_dataset, val_dataset = load_dataset(img_folder_name, validation_split, seed, 
                 image_size, crop_to_aspect_ratio, 
                 pad_to_aspect_ratio)
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset, dataset_repetitions,
                    batch_size)

    # Create and compile the model
    model = DiffusionModel(image_size, widths, block_depth, eta)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        # Loss function: Pixelwise mean absolute error (MAE).
        loss=keras.losses.mean_absolute_error,
    )

    # Calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(train_dataset)

    # Load the model if desired
    if load_and_train: 
        model.load_weights(checkpoint_path)

    # Copy parameters file into the model folder
    shutil.copy("parameters.py", f"{folder_path}/")
    print(f"\nParameters copied to {folder_path}/")

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=num_epochs,
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
    save_history(history, folder_path)

    # Perform inference when training finishes
    InferenceDiffusionModel()

def InferenceDiffusionModel(): 
    # Load and prepare the dataset
    train_dataset, val_dataset = load_dataset(img_folder_name, validation_split, seed, 
                 image_size, crop_to_aspect_ratio, 
                 pad_to_aspect_ratio)
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset, dataset_repetitions,
                    batch_size)

    # Build and load the model
    model = DiffusionModel(image_size, widths, block_depth, eta)
    model.normalizer.adapt(train_dataset) 
    model.load_weights(checkpoint_path)

    # Generate the images
    generated_images = model.generate(images_to_generate, generate_diffusion_steps, True)

    # Create directory in model's folder and save the images
    generated_dir = os.path.join(folder_path, "generated_images")
    if not os.path.exists(generated_dir): 
        os.makedirs(generated_dir)

    # Get today's date
    current_date = datetime.datetime.now() 
    formatted_date = current_date.strftime("%m%d%y")

    # Save the images in the folder with the 
    # corresponding naming convention
    index = 1
    for image in generated_images: 
        image_name = f"S0_D{formatted_date}_{formatted_date}_0_{index}_DM_AUG_{label}.JPG"
        tf.keras.preprocessing.image.save_img(f"{generated_dir}/{image_name}", image) 
        index = index + 1

def ContextualInpainting(): 
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

    # Load and prepare the datasets
    train_dataset, val_dataset = load_dataset(img_folder_name, validation_split, seed, 
                 image_size, crop_to_aspect_ratio, 
                 pad_to_aspect_ratio)
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset, dataset_repetitions,
                    batch_size)

    # Build and load the model
    model = DiffusionModel(image_size, widths, block_depth, eta)
    model.normalizer.adapt(train_dataset) 
    model.load_weights(checkpoint_path)

    # Load masks and images
    """MODIFY TO HANDLE LONG LISTS OF IMAGES AND MASKS"""
    image_list, mask_list, image_dir, mask_dir = load_inpainting_data_temp(inpainting_dir)
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
    inpainted_dir = os.path.join(folder_path, "inpainted_images")
    if not os.path.exists(inpainted_dir): 
        os.makedirs(inpainted_dir)

    """MODIFY THE NAMING CONVENTION TO FIT INTO THE PIPELINE""" 
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    for image in inpainted_images: 
        tf.keras.preprocessing.image.save_img(f"{inpainted_dir}/{timestamp}_inpainted_img_{image_name}.jpg", image) 

if __name__ == "__main__":
    """ 
    ENSURE WE ARE USING THE CORRECT VERSION 
    ONLY KERAS 3.6 AND TENSORFLOW 2.16.1 HAS 
    BEEN PROVEN TO WORK WITH THIS SCRIPT
    """
    print(keras.__version__)
    print(tf.__version__)

    print("\nIf Keras version is not 3.6.0, and Tensorflow 2.16.1 or 2.17.0 you might run into issues\n")

    # Set backend to Tensorflow
    os.environ["KERAS_BACKEND"] = "tensorflow"

    # Ensure that the model does not take all the GPU memory
    gpus = tf.config.list_physical_devices("GPU")
    if gpus: 
        try: 
            for gpu in gpus: 
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e: print(e)

    # Only error messages will be displayed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Use mixed_precision if dictated
    if used_mix_precision: 
        mixed_precision.set_global_policy("mixed_float16")
        print("\nUsing mixed_precision float16\n")
    else: 
        mixed_precision.set_global_policy("float32")
        print("\nDefault, using float32\n")
        
    if runtime == "training":
        TrainDiffusionModel()
    elif runtime == "inference":
        if not os.path.exists(folder_path): 
            raise Exception("\nWARNING: Cannot find the directory where the model and all its files are stored\n")
        InferenceDiffusionModel()
        print(f"\n Finish generating {images_to_generate} images\n")
    elif runtime == "inpaint":
        if not os.path.exists(folder_path): 
            raise Exception("\nWARNING: Cannot find the directory where the model and all its files are stored\n")
        """MODIFY THIS LATER ON"""
        ContextualInpainting()
        print(f"\nFinish inpainting images\n")

   









