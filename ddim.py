""" 
THIS DDIM MODEL
TRAINS OUR 100X300 IMAGES

ONLY TENSORFLOW 2.16.1 WILL 
WORK, PLEASE REFER TO THE
OFFICIAL KERAS WEBSITE
https://keras.io/getting_started/
"""

""" ALL IMPORTS """
# import necessary libraries
import os
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
import cv2 as cv
import logging
from datetime import datetime
import shutil

# import from local scripts
from diffusion_model import DiffusionModel
from callbacks import * 
from parameters import *

""" HELPER FUNCTIONS """
def load_dataset(): 
    """
    Loads the dataset for training
    """
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, img_folder_name)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        img_dir, 
        validation_split = validation_split,
        subset="training", 
        seed = seed,
        image_size = (image_size[0], image_size[1]),  
        batch_size = None,
        shuffle = True,
        crop_to_aspect_ratio = crop_to_aspect_ratio,
        pad_to_aspect_ratio = pad_to_aspect_ratio,
        #labels = labels, 
        #label_mode = None
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        img_dir, 
        validation_split = validation_split,
        subset="validation", 
        seed = seed,
        image_size = (image_size[0], image_size[1]), 
        batch_size = None,
        shuffle = True,
        crop_to_aspect_ratio = crop_to_aspect_ratio,
        pad_to_aspect_ratio = pad_to_aspect_ratio,
        #labels = labels, 
        #label_mode = None
    )
    return train_ds, val_ds


def prepare_dataset(train_ds, val_ds): 
    """
    Prepares the dataset for training, used in combination with load_dataset
    """
    train_ds = (train_ds
        .map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE) # each dataset has the structure
        .cache()                                                   # (image, labels) when inputting to 
        .repeat(dataset_repetitions)                               # map
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE))
    val_ds = (val_ds
        .map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(dataset_repetitions)
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)) # THIS IS A PREFETCH DATASET
    return train_ds, val_ds

def normalize_image(images, _):    
    # clip pixel values to the range [0, 1]
    return tf.clip_by_value(images / 255, 0.0, 1.0)

def plot_images(dataset, num_images=5):
    # Create an iterator to get images from the dataset
    iterator = iter(dataset)
    
    # Get the first batch of images
    images = next(iterator)

    # Plot the images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].axis("off")
    plt.show()

# def load_inpainting_data_temp(): 
#     """
#     TEMPORARY FUNCTION. DO NOT USE
#     """
#     cwd = os.getcwd()
#     mask_and_image_dir = os.path.join(cwd, inpainting_dir)
#     mask_and_image_dir_list = os.listdir(mask_and_image_dir)

#     # identify directories for images and masks
#     if mask_and_image_dir_list[0] == "masks": 
#         mask_dir = os.path.join(mask_and_image_dir, os.listdir(mask_and_image_dir)[0])
#         image_dir = os.path.join(mask_and_image_dir, os.listdir(mask_and_image_dir)[1])
#     else: 
#         mask_dir = os.path.join(mask_and_image_dir, os.listdir(mask_and_image_dir)[1])
#         image_dir = os.path.join(mask_and_image_dir, os.listdir(mask_and_image_dir)[0])

#     # sort images and masks dir list
#     mask_dir_list = sorted(os.listdir(mask_dir))
#     image_dir_list = sorted(os.listdir(image_dir))

# def save_history(history, dict_key_list): 
#     for key in dict_key_list: 
#         plt.plot(history.history[key], label=key)

#     plt.title("Training losses")
#     plt.ylabel("Value")
#     plt.xlabel("Epochs")

#     plt.legend(loc="upper left")
#     plt.savefig(f"{folder_path}/training_loss.png")
#     plt.close()

def save_history(history):
    # Saves the plot of the history
    for key in history.history.keys():
        plt.plot(history.history[key], label=key)

    plt.title("Training losses and metrics")
    plt.ylabel("Value")
    plt.xlabel("Epochs")
    plt.legend(loc="upper left")
    
    # Save the plot
    plt.savefig(f"{folder_path}/training_loss.png")
    plt.close()    

""" Main Runtime """
def TrainDiffusionModel():
    # Load and prepare the dataset
    train_dataset, val_dataset = load_dataset()
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset)

    # Create and compile the model
    model = DiffusionModel(image_size, widths, block_depth, eta)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        loss=keras.losses.mean_absolute_error,
        # Loss function: Pixelwise mean absolute error (MAE).
    )

    # Calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(train_dataset)

    if load_and_train: 
        model.load_weights(checkpoint_path)

    # Copy parameters file into the model folder
    shutil.copy("parameters.py", f"{folder_path}/")
    print(f"Parameters copied to {folder_path}/\n")

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
    save_history(history)

    # Perform inference
    InferenceDiffusionModel()

def InferenceDiffusionModel(): 
    # Load and prepare the dataset
    train_dataset, val_dataset = load_dataset()
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset)

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
    train_dataset, val_dataset = load_dataset()
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset)

    # Build and load the model
    model = DiffusionModel(image_size, widths, block_depth, eta)
    model.normalizer.adapt(train_dataset) 
    model.load_weights(checkpoint_path)

    # Load masks and images
    """MODIFY TO HANDLE LONG LISTS OF IMAGES AND MASKS"""
    image_list, mask_list, image_dir, mask_dir = load_inpainting_data_temp()
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

    # Show if GPU is being used
    print(f"\nNum GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\n")

    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    if runtime == "training":
        logging.basicConfig(level=logging.INFO, filename=f"{folder_path}/{folder_path}error.log")
        TrainDiffusionModel()
    elif runtime == "inference":
        if not os.path.exists(folder_path): 
            raise Exception("\nWARNING: Cannot find the directory where the model and all its files are stored\n")
        InferenceDiffusionModel()
        print(f"\n Finish generating {images_to_generate} images\n")
    elif runtime == "inpaint":
        if not os.path.exists(folder_path): 
            raise Exception("\nWARNING: Cannot find the directory where the model and all its files are stored\n")
        ContextualInpainting()
        """MODIFY THIS LATER ON"""
        print(f"\nFinish inpainting images\n")

   









