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
import datetime
import shutil

# Import from local script
from diffusion_model import DiffusionModel
from config import Config
from utils import *

""" 
This callback will generate an image after a
certain amount of epochs. 
"""
class GenerateOnEpoch(keras.callbacks.Callback): 
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

def PrepareData_LoadModel(
                        dataset_name, validation_split, seed, 
                        image_size, batch_size, 
                          
                        widths, block_depth, eta, max_signal_rate, min_signal_rate, 
                        ema, embedding_dims, attention_in_up_down_sample, attention_in_bottleneck): 

    # Load and prepare the dataset
    train_dataset, val_dataset = load_dataset(dataset_name, validation_split, seed, 
                 image_size)
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset, batch_size)
    
    # Create and compile the model
    model = DiffusionModel(image_size, widths, block_depth, eta, max_signal_rate, min_signal_rate, batch_size, 
                 ema, embedding_dims, attention_in_up_down_sample, attention_in_bottleneck)

    # Calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(train_dataset)

    return train_dataset, val_dataset, model

def GetCallbacks(generate_on_epoch, generate_diffusion_steps,
                 folder_path, checkpoint_monitor, early_stop_monitor, 
                 early_stop_min_delta, early_stop_patience, early_stop_start_epoch): 

    # Generates an image per amount of epoch
    generate_image_callback = GenerateOnEpoch(generate_on_epoch, generate_diffusion_steps, folder_path)

    # Last checkpoint callback
    last_checkpoint_path = f"{folder_path}/last_diffusion_model.weights.h5"
    last_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=last_checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_best_only=False,)
    
    # Save the best performing models
    checkpoint_path = f"{folder_path}/best_diffusion_model.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor=checkpoint_monitor,
        verbose=1,
        mode="min",
        save_best_only=True,)

    # Early stopping
    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor=early_stop_monitor, 
        min_delta=early_stop_min_delta,
        patience=early_stop_patience,
        verbose=1,
        mode="min",
        restore_best_weights=True,
        start_from_epoch=early_stop_start_epoch,
    )

    # Log losses
    current_time = datetime.datetime.now() 
    formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    csv_callback = keras.callbacks.CSVLogger(filename=f"{folder_path}/csv_log_{formatted_time}.csv", separator=",", append=True)

    return generate_image_callback, last_checkpoint_callback, checkpoint_callback, early_stop_callback, csv_callback

"""IN THE FUTURE MAKES THIS MORE READABLE"""
def TrainDiffusionModel(use_mix_precision, model,
                        learning_rate, weight_decay, num_epochs, 
                        load_and_train, folder_path, 
                        train_dataset, val_dataset, 

                        checkpoint_path,
                        generate_image_callback, last_checkpoint_callback, 
                        checkpoint_callback, early_stop_callback, csv_callback):

    # Implement mixed_precision if dictated
    if use_mix_precision: 
        mixed_precision.set_global_policy("mixed_float16")
        print("\nUsing mixed_precision float16\n")
    else: 
        mixed_precision.set_global_policy("float32")
        print("\nDefault, using float32\n")

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        # Loss function: Pixelwise mean absolute error (MAE).
        loss=keras.losses.mean_absolute_error,
    )

    # Load the model if desired
    if load_and_train: 
        model.load_weights(checkpoint_path)

    # Copy parameters file into the model folder
    shutil.copy("config.py", f"{folder_path}/")
    print(f"\nConfig copied to {folder_path}/")

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
    
def InferenceDiffusionModel(model, checkpoint_path, folder_path, 
                            images_to_generate, generate_diffusion_steps, label): 

    # Load the model's weights
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

def ContextualInpainting(model, checkpoint_path, folder_path, inpainting_dir): 
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
    
    There are two ways to train the model. One way, is to run
    the run_diffusion.py script, after setting 'runtime' to 'training'
    in config.py. The other way is using this train.py file. I recommended
    running this script, because it abstract the code from run_diffusion, and
    we can train 3-6 models at once with this training loop, whereas in 
    run_diffusion.py, you can only train one model. 

    ------------------------------------------
    - How to Use -

    You can either modify parameters in config.py and run 'run_diffusion'
    to run single models, or if you want to train 3-6 models in a training loop, 
    use the argparse parameters below. 

    """





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
    print(f"\nThese are the current GPUs detected in the system:\n {gpus}\n")

    # Obtain the config file
    config = Config()

    if gpus: 
        try: 
            tf.config.set_visible_devices(gpus[config.gpu_index], "GPU")
            print(f"Currently using {gpus[config.gpu_index]}\n")

            tf.config.experimental.set_memory_growth(gpus[config.gpu_index], True)
            print(f"Set to growing memory\n")
        except RuntimeError as e: print(e)

    # Only error messages will be displayed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Prepare Dataset and Load Model
    train_dataset, val_dataset, model = PrepareData_LoadModel(
                        config.dataset_name, config.validation_split, config.seed,                         
                        config.image_size, config.batch_size, 
                          
                        config.widths, config.block_depth, config.eta, config.max_signal_rate, config.min_signal_rate, 
                        config.ema, config.embedding_dims, config.attention_in_up_down_sample, config.attention_in_bottleneck)

    checkpoint_path = f"{config.folder_path}/best_diffusion_model.weights.h5"

    if config.runtime == "training":
        # Get callbacks
        generate_image_callback, last_checkpoint_callback, checkpoint_callback, early_stop_callback, csv_callback = GetCallbacks(
            config.generate_on_epoch, config.generate_diffusion_steps,
                 config.folder_path, config.checkpoint_monitor, config.early_stop_monitor, 
                 config.early_stop_min_delta, config.early_stop_patience, config.early_stop_start_epoch
        )

        # Train the Diffusion Model
        TrainDiffusionModel(config.use_mix_precision, model, 
                            config.learning_rate, config.weigth_decay, config.num_epochs, 
                            config.load_and_train, config.folder_path, 
                            train_dataset, val_dataset, 
                        
                            checkpoint_path,
                            generate_image_callback, last_checkpoint_callback, 
                            checkpoint_callback, early_stop_callback, csv_callback)
    
        # Perform Inference after training
        if not config.use_mix_precision: 
            mixed_precision.set_global_policy("mixed_float16")
            print("\nUsing mixed_precision float16\n")
        else: 
            mixed_precision.set_global_policy("float32")
            print("\nDefault, using float32\n")

        InferenceDiffusionModel(model, checkpoint_path, config.folder_path, 
                                    config.images_to_generate, config.generate_diffusion_steps, config.label)        

        print(f"\nFinish training for {config.num_epochs}. Check {config.folder_path} for the model")
                            
    elif config.runtime == "inference":
        if not os.path.exists(config.folder_path): 
            raise Exception("\nWARNING: Cannot find the directory where the model and all its files are stored\n")
        
        InferenceDiffusionModel(model, checkpoint_path, config.folder_path, 
                                    config.images_to_generate, config.generate_diffusion_steps, config.label)        
        print(f"\n Finish generating {config.images_to_generate} images\n")
    elif config.runtime == "inpaint":
        if not os.path.exists(config.folder_path): 
            raise Exception("\nWARNING: Cannot find the directory where the model and all its files are stored\n")
                
        """MODIFY THIS LATER ON"""
        ContextualInpainting(model, checkpoint_path, config.folder_path, config.inpainting_dir)
        print(f"\nFinish inpainting images\n")
