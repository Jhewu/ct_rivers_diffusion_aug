"""
This utility script stores all of the 
helper functions used in run_diffusion. 
"""

# Import necessary libraries
import os
import matplotlib.pyplot as plt 
import tensorflow as tf
import argparse
import datetime
import keras
import pickle
import cv2 as cv
from math import ceil, floor
import shutil


"""
PARSING ARGUMENTS----------------------------------------------------------------------------------------------------------------------------------------------
"""

def ParseArgs(des): 
    """
    This function parses all of the arguments
    """
    parser = argparse.ArgumentParser(description=des.lstrip(" "),formatter_class=argparse.RawTextHelpFormatter)

    # General parameters
    parser.add_argument('--in_dir', type=str, help="The dir that contains the label folder with river images, structured similarly like this: 'L1/1/image.JPG'")
    parser.add_argument('--out_dir', type=str, help="The dir to save model weight, callbacks and results")

    # Training parameters
    parser.add_argument('--runtime', type=str, choices=["training", "inference", "inpainting"], help="Run mode: training or inference or inpainting")
    parser.add_argument('--load_and_train', action='store_true', help="Whether to load a pre-trained model to train")
    parser.add_argument('--eta', type=float, help="Eta parameter for noise scheduling")
    parser.add_argument('--image_size', type=tuple, help="Size of the input images (height, width)")

    # Optimization (Training) parameters
    parser.add_argument('--num_epochs', type=int, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, help="Learning rate")
    parser.add_argument('--use_mix_precision', action='store_true', help="Whether to use mixed precision training")    
    parser.add_argument('--gpu_index', type=int, help="Index of the GPU to use")

    # U-Net architecture parameters
    parser.add_argument('--embedding_dims', type=int, help="Dimensions for embeddings")
    parser.add_argument('--widths', type=int, nargs='+', help="Widths for each convolutional layer")
    parser.add_argument('--block_depth', type=int, help="Depth of the U-Net blocks")
    parser.add_argument('--attention_in_bottleneck', action='store_true', help="Whether to use attention in bottleneck")
    parser.add_argument('--attention_in_up_down_sample', action='store_true', help="Whether to use attention in up/down sampling layers")

    # Inference parameters
    parser.add_argument('--model_dir', type=str, help="The dir where the model's weight is located in")
    parser.add_argument('--images_to_generate', type=int, help="Number of images to generate during inference")
    parser.add_argument('--generate_diffusion_steps', type=int, help="Number of diffusion steps for generation")

    # Running from subprocess
    # This is to check if we are running from a subprocess call
    # If we are instead of saving the images on the model_dir, 
    # We save it to the respective out_dir provided by the subprocess call
    parser.add_argument('--subprocess', action='store_true', help="Boolean flag to indicate running in subprocess")

    # Parse arguments
    return parser.parse_args()

def AssignArgs(config, args): 
    """
    This function runs the diffusion loop. It creates the parse arguments by extending 
    the list, and then pass it to run_diffusion.py. The function also contains a checker
    to load the config_file.pkl when necessary (during load and train and during inference)
    """

    # General hyperparameters
    if args.in_dir is not None:
        config.in_dir = args.in_dir
    if args.out_dir is not None:
        config.out_dir = args.out_dir
    if args.runtime is not None:
        config.runtime = args.runtime
    if args.load_and_train is not None:
        config.load_and_train = args.load_and_train
    if args.eta is not None:
        config.eta = args.eta
    if args.image_size is not None:
        config.image_size = args.image_size

    # Optimization (Training)
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.use_mix_precision is not None:
        config.use_mix_precision = args.use_mix_precision
    if args.gpu_index is not None:
        config.gpu_index = args.gpu_index

    # U-Net architecture
    if args.embedding_dims is not None:
        config.embedding_dims = args.embedding_dims
    if args.widths is not None:
        config.widths = args.widths
    if args.block_depth is not None:
        config.block_depth = args.block_depth
    if args.attention_in_bottleneck is not None:
        config.attention_in_bottleneck = args.attention_in_bottleneck
    if args.attention_in_up_down_sample is not None:
        config.attention_in_up_down_sample = args.attention_in_up_down_sample

    # Inference hyperparameters
    if args.model_dir is not None:
        config.model_dir = args.model_dir
    if args.images_to_generate is not None:
        config.images_to_generate = args.images_to_generate
    if args.generate_diffusion_steps is not None:
        config.generate_diffusion_steps = args.generate_diffusion_steps

    # Running from subprocess
    if args.subprocess:
        config.subprocess = True
    else:
        config.subprocess = False

    return config

def LoadPrevConfig(config, config_path): 
    if config.load_and_train or config.runtime == "inference": 
        print(f"\nLoaded config_file.pkl\n")
        # Load config_file.pkl here so we can easily restore previous parameters
        prev_config = load_config_from_pickle(config_path)

        # Change the architecture hyperparameters
        config.embedding_dims = prev_config.embedding_dims
        config.widths = prev_config.widths
        config.block_depth = prev_config.block_depth
        config.attention_in_bottleneck = prev_config.attention_in_bottleneck
        config.attention_in_up_down_sample = prev_config.attention_in_up_down_sample

        # Change preprocessing hyperparameters
        config.seed = prev_config.seed
        config.validation_split = prev_config.validation_split
        config.image_size = prev_config.image_size

        # Change optimization
        config.min_signal_rate = prev_config.min_signal_rate
        config.max_signal_rate = prev_config.max_signal_rate
    else: 
        print(f"\nDid not load config_file.pkl\n")
   
    return config

"""
KERAS CALLBACKS----------------------------------------------------------------------------------------------------------------------------------------------
"""

class GenerateOnEpoch(keras.callbacks.Callback): 
    """ 
    This callback will generate an image after a
    certain amount of epochs. 
    """
    def __init__(self, generate_on_epoch, generate_diffusion_steps, folder_path, **kwargs):
        super().__init__(**kwargs)
        self.generate_on_epoch = generate_on_epoch
        self.generate_diffusion_steps = generate_diffusion_steps
        self.folder_path = folder_path

    def on_epoch_end(self, epoch, logs=None): 
        if (epoch + 1) % self.generate_on_epoch == 0: 
            # Generate only one image
            generated_images = self.model.generate(1, self.generate_diffusion_steps, True)
            # Get current time
            current_time = datetime.datetime.now() 
            formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
            # Create a new directory
            generated_dir = os.path.join(self.folder_path, "plot_images")
            if not os.path.exists(generated_dir): 
                os.makedirs(generated_dir)
            # Save the generated images
            index = 1
            for image in generated_images: 
                tf.keras.preprocessing.image.save_img(f"{generated_dir}/{formatted_time}_generated_img_{index}.jpg", image) 
                index = index + 1

def GetCallbacks(config): 
    """
    Obtain callbacks to run during training
    """
    # Generates an image per amount of epoch
    generate_image_callback = GenerateOnEpoch(config.generate_on_epoch, config.generate_diffusion_steps, config.out_dir)

    # If it's not load and train, we set model_dir to out_dir (for Inference later)
    # However, if we are load and train, then we do not set the out_dir to be the same
    if config.load_and_train: 
        config.out_dir = config.model_dir
    else:
        config.model_dir = config.out_dir


    # Last checkpoint callback
    last_checkpoint_path = f"{config.out_dir}/last_diffusion_model.weights.h5"
    last_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=last_checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_best_only=False,)
    
    # Save the best performing models
    checkpoint_path = f"{config.out_dir}/best_diffusion_model.weights.h5"

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor=config.checkpoint_monitor,
        verbose=1,
        mode="min",
        save_best_only=True,)

    # Early stopping
    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor=config.early_stop_monitor, 
        min_delta=config.early_stop_min_delta,
        patience=config.early_stop_patience,
        verbose=1,
        mode="min",
        restore_best_weights=True,
        start_from_epoch=config.early_stop_start_epoch,
    )

    # Log losses
    current_time = datetime.datetime.now() 
    formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    csv_callback = keras.callbacks.CSVLogger(filename=f"{config.model_dir}/csv_log_{formatted_time}.csv", separator=",", append=True)

    return generate_image_callback, last_checkpoint_callback, checkpoint_callback, early_stop_callback, csv_callback


"""
RUN_DIFFUSION HELPER FUNCTIONS------------------------------------------------------------------------------------------------------------------------------------
"""

def log_config_parameters(config, log_file_path):
    """
    Logs all parameters from the config object into a log file.
    """
    # Create the log file directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Open the log file in append mode
    with open(log_file_path, 'a') as log_file:
        log_file.write("Config Parameters:\n")
        log_file.write("="*30 + "\n")
        
        # If the config is an object with __dict__, log the attributes
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = config  # Assume config is a dictionary
        
        # Log each parameter key-value pair
        for key, value in config_dict.items():
            log_file.write(f"{key}: {value}\n")
        
        log_file.write("\n")

def save_config_to_pickle(config, pickle_file_path):
    """
    Saves the config object to a pickle file.
    """
    # Open the pickle file in binary write mode
    with open(pickle_file_path, 'wb') as pickle_file:
        # Serialize the config object and save it to the file
        pickle.dump(config, pickle_file)
    
    print(f"Config saved to pickle file: {pickle_file_path}")

def load_config_from_pickle(pickle_file_path):
    """
    Loads the config object from a pickle file.
    """
    # Open the pickle file in binary read mode
    with open(pickle_file_path, 'rb') as pickle_file:
        # Deserialize the config object from the file
        config = pickle.load(pickle_file)
    return config

def load_dataset(img_folder_name, validation_split, seed, 
                 image_size): 
    """
    Loads the dataset for training
    """
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, img_folder_name)

    def load_data(split): 
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            img_dir, 
            validation_split = validation_split,
            subset= split,
            seed = seed,
            image_size = (image_size[0], image_size[1]),  
            batch_size = None,
            shuffle = True,
            crop_to_aspect_ratio = True,
            pad_to_aspect_ratio = False,)
        return dataset

    if validation_split > 0:     
        train_ds = load_data("training")
        val_ds = load_data("validation")
    else: 
        train_ds = load_data(None)
        val_ds = None
    
    return train_ds, val_ds

def prepare_dataset(train_ds, val_ds, batch_size): 
    """
    Prepares the dataset for training, used in combination with load_dataset
    """
    def prepare(ds): 
        ds = (ds
            .map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE) # each dataset has the structure
            .cache()                                                   # (image, labels) when inputting to 
            .shuffle(10 * batch_size)
            .batch(batch_size, drop_remainder=True)
            .prefetch(buffer_size=tf.data.AUTOTUNE))
        return ds
    
    if val_ds == None:     
        train_ds = prepare(train_ds)
    else: 
        train_ds = prepare(train_ds)
        val_ds = prepare(val_ds)
        
    return train_ds, val_ds

def normalize_image(images, _):    
    """
    Clip pixel values to the range [0, 1]. 
    Used in prepare_dataset
    """
    return tf.clip_by_value(images / 255, 0.0, 1.0)

def load_inpainting_data_temp(mask_and_image_dir): 
    """
    UPDATE FUNCTION IN THE FUTURE
    """
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
    """
    Saves the plot of the history
    """
    for key in history.history.keys():
        plt.plot(history.history[key], label=key)

    plt.title("Training losses and metrics")
    plt.ylabel("Value")
    plt.xlabel("Epochs")
    plt.legend(loc="upper left")
    
    # Save the plot
    plt.savefig(f"{folder_path}/training_loss.png")
    plt.close()    

def CreateDir(folder_name):
   """
   Create directory if there's none
   """
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

"""
SITE_ID_BALANCER HELPER FUNCTIONS--------------------------------------------------------------------------------------------------------------------------------------
"""

def Data_augmentation(img, THETA, FACT, flipped):
   """
   Main data augmentation function 
   """
   img = cv.imread(img)
   if flipped == False: # if regular rotation
       pad = Pad_img(img)   
       rot = Rotate_img(pad, THETA)
       up = Upsample_img(rot, FACT)
       return Center_crop(up, img)
   else:                # if horizontal flip rotation
       flip = Flip_img(img)
       pad = Pad_img(flip)   
       rot = Rotate_img(pad, THETA)
       up = Upsample_img(rot, FACT)
       return Center_crop(up, img)
   
def Pad_img(img):
   row, col, colors = img.shape
   padding_lr = floor(col/2) # left and right
   padding_tb = floor(row/2) # top and bottom
   return cv.copyMakeBorder(img, padding_tb, padding_tb,
                       padding_lr, padding_lr, borderType = cv.BORDER_CONSTANT, value = (0, 0,0))
def Flip_img(img):
   return cv.flip(img, 1)

def Get_center(coord):
   return ceil((coord-1)/2.0)

def Rotate_img(img, THETA):
   row, col, colors = img.shape
   centerx = Get_center(col)
   centery = Get_center(row)
   matrix = cv.getRotationMatrix2D((centerx, centery), THETA, 1)
   return cv.warpAffine(img, matrix, (col,row))

def Upsample_img(img, FACT):
   # for each 5 degrees, increase fact by 0.3x
   return cv.resize(img, None, fx=FACT, fy=FACT, interpolation = cv.INTER_CUBIC)

def Center_crop(img, og_img):
   row, col, color = img.shape
   og_row, og_col, og_color = og_img.shape
   centerx = Get_center(col)
   centery = Get_center(row) # --> padded, rotated and upscaled image center
   ogx = Get_center(og_col)
   ogy = Get_center(og_row) # ---> image center of original image
   return img[centery-ogy:centery+ogy, centerx-ogx:centerx+ogx]

def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def SortDict(dict):
   # Sort the dictionary by its value in ascending order
   sorted_items = sorted(dict.items(), key=lambda item: item[1])
   return sorted_items

def Get_center(coord):
   return ceil((coord-1)/2.0)

def Copy_dir(src, dst): 
   try: 
      shutil.copytree(src, dst)
   except Exception as e:
      print(f"Error copying {src} to {dst}: {e}")
      print("This does not mean the program is not working correctly, it just means that the directory already exists\n")