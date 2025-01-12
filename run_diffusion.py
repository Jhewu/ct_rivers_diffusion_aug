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
import argparse

# Import from local script
from diffusion_model import DiffusionModel
from config import Config
from utils import *

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

def GetCallbacks(config): 
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
    - Training the Diffusion Model -

    There are two ways to train the model. One way is to train 
    a single model which you can do by 'python3 run_diffusion.py'
    The other way is to train multiple models in a training loop, 
    which you can do by 'python3 run_training_loop.py'

    There is also two ways to modify the parameters for the model. 
    The easiest way is to modify the config.py file which contains all the 
    script's parameters. The other way is to use argparse, when running either
    'run_diffusion.py' or 'run_training_loop.py' , however not all parameters 
    are available for argparse. 

    ------------------------------------------
    - How to Use -

    Please refer to the README.md file for more information on how to use
    or access help, by 'python3 run_diffusion.py -h'

    ------------------------------------------
    """
    
    """-----ARGPARSE BELOW------------------------------------------------------------------------------------"""
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
    args = parser.parse_args()
    """-----ARGPARSE ABOVE------------------------------------------------------------------------------------"""

    """-----ASSIGN ARGS BELOW---------------------------------------------------------------------------------"""
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
    """-----ASSIGN ARGS ABOVE---------------------------------------------------------------------------------"""

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
