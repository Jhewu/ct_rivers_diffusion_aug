"""
This utilities script stores all of the 
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
    parser.add_argument('--load_and_train', type=bool, help="Whether to load a pre-trained model to train")
    parser.add_argument('--eta', type=float, help="Eta parameter for noise scheduling")
    parser.add_argument('--image_size', type=tuple, help="Size of the input images (height, width)")

    # Optimization (Training) parameters
    parser.add_argument('--num_epochs', type=int, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, help="Learning rate")
    parser.add_argument('--use_mix_precision', type=bool, help="Whether to use mixed precision training")
    parser.add_argument('--gpu_index', type=int, help="Index of the GPU to use")

    # U-Net architecture parameters
    parser.add_argument('--embedding_dims', type=int, help="Dimensions for embeddings")
    parser.add_argument('--widths', type=int, nargs='+', help="Widths for each convolutional layer")
    parser.add_argument('--block_depth', type=int, help="Depth of the U-Net blocks")
    parser.add_argument('--attention_in_bottleneck', type=bool, help="Whether to use attention in bottleneck")
    parser.add_argument('--attention_in_up_down_sample', type=bool, help="Whether to use attention in up/down sampling layers")

    # Inference parameters
    parser.add_argument('--model_dir', type=str, help="The dir where the model's weight is located in")
    parser.add_argument('--images_to_generate', type=int, help="Number of images to generate during inference")
    parser.add_argument('--generate_diffusion_steps', type=int, help="Number of diffusion steps for generation")

    # Parse arguments
    return parser.parse_args()

def AssignArgs(config, args): 
    """
    This function runs the diffusion loop. It creates the parse arguments by extending 
    the list, and then pass it to run_diffusion.py
    """
    # General parameters
    if args.in_dir is not None:
        config.in_dir = args.in_dir
    if args.out_dir is not None:
        config.out_dir = args.out_dir

    # Training parameters
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

    # Inference parameters
    if args.model_dir is not None:
        config.model_dir = args.model_dir
    if args.images_to_generate is not None:
        config.images_to_generate = args.images_to_generate
    if args.generate_diffusion_steps is not None:
        config.generate_diffusion_steps = args.generate_diffusion_steps


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

    # Last checkpoint callback
    last_checkpoint_path = f"{config.out_dir}/last_diffusion_model.weights.h5"
    last_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=last_checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_best_only=False,)
    
    # Save the best performing models
    checkpoint_path = f"{config.out_dir}/best_diffusion_model.weights.h5"
    config.model_dir = config.out_dir
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
    csv_callback = keras.callbacks.CSVLogger(filename=f"{config.out_dir}/csv_log_{formatted_time}.csv", separator=",", append=True)

    return generate_image_callback, last_checkpoint_callback, checkpoint_callback, early_stop_callback, csv_callback


"""

HELPER FUNCTIONS----------------------------------------------------------------------------------------------------------------------------------------------

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