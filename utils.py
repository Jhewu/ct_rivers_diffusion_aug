"""
This utilities script stores all of the 
helper functions used in run_diffusion. 
"""

# Import necessary libraries
import os
import matplotlib.pyplot as plt 
import tensorflow as tf

""" HELPER FUNCTIONS """
def load_dataset(img_folder_name, validation_split, seed, 
                 image_size): 
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
        crop_to_aspect_ratio = True,
        pad_to_aspect_ratio = False,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        img_dir, 
        validation_split = validation_split,
        subset="validation", 
        seed = seed,
        image_size = (image_size[0], image_size[1]), 
        batch_size = None,
        shuffle = True,
        crop_to_aspect_ratio = True,
        pad_to_aspect_ratio = False,
    )

    return train_ds, val_ds

def prepare_dataset(train_ds, val_ds, batch_size): 
    """
    Prepares the dataset for training, used in combination with load_dataset
    """
    train_ds = (train_ds
        .map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE) # each dataset has the structure
        .cache()                                                   # (image, labels) when inputting to 
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE))
    val_ds = (val_ds
        .map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)) # THIS IS A PREFETCH DATASET

    return train_ds, val_ds

def normalize_image(images, _):    
    """
    Clip pixel values to the range [0, 1]. 
    Used in prepare_dataset
    """
    return tf.clip_by_value(images / 255, 0.0, 1.0)

def load_inpainting_data_temp(mask_and_image_dir): 
    cwd = os.getcwd()
    mask_and_image_dir = os.path.join(cwd, mask_and_image_dir)
    mask_and_image_dir_list = os.listdir(mask_and_image_dir)

    # Identify directories for images and masks
    if mask_and_image_dir_list[0] == "masks": 
        mask_dir = os.path.join(mask_and_image_dir, os.listdir(mask_and_image_dir)[0])
        image_dir = os.path.join(mask_and_image_dir, os.listdir(mask_and_image_dir)[1])
    else: 
        mask_dir = os.path.join(mask_and_image_dir, os.listdir(mask_and_image_dir)[1])
        image_dir = os.path.join(mask_and_image_dir, os.listdir(mask_and_image_dir)[0])

    # Sort images and masks dir list
    mask_dir_list = sorted(os.listdir(mask_dir))
    image_dir_list = sorted(os.listdir(image_dir))
        
    return image_dir_list, mask_dir_list, image_dir, mask_dir

def save_history(history, folder_path):
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