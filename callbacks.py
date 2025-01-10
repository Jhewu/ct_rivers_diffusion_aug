"""
THIS SCRIPT CONTAINS ALL THE CALLBACK FUNCTIONS 
FOR THE DIFFUSION MODEL
"""

# Import necessary libraries
import keras
import csv
import tensorflow as tf
import numpy
import datetime

# Import form local script
from parameters import folder_path, checkpoint_monitor, early_stop_monitor, early_stop_patience, early_stop_start_epoch, early_stop_min_delta, generate_on_epoch

# Get current time
current_time = datetime.datetime.now() 
formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")

""" 
Create Custom Callback 
This callback will generate an image after a
certain amount of epochs. 
"""
class CustomCallback(keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs=None): 
        if (epoch + 1) % generate_on_epoch == 0: 
            # Generate only one image
            generated_images = self.model.generate(1, generate_diffusion_steps, True)
            # Get current time
            current_time = datetime.datetime.now() 
            formatted_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
            # Create a new directory
            generated_dir = os.path.join(folder_path, "plot_images")
            if not os.path.exists(generated_dir): 
                os.makedirs(generated_dir)
            # Save the generated images
            index = 1
            for image in generated_images: 
                image_name = f"S0_D{formatted_time}_{formatted_time}_0_{index}_DM_AUG_{label}.JPG"
                tf.keras.preprocessing.image.save_img(f"{generated_dir}/{formatted_time}_generated_img_{index}.jpg", image) 
                index = index + 1

generate_image_callback = CustomCallback()

"""
Last Checkpoint Callback
Save the last model
"""
last_checkpoint_path = f"{folder_path}/last_diffusion_model.weights.h5"
last_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=last_checkpoint_path,
    save_weights_only=True,
    verbose=1,
    save_best_only=False,)

"""
Checkpoint Callback
Save the best performing models
only
"""
checkpoint_path = f"{folder_path}/diffusion_model.weights.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor=checkpoint_monitor,
    verbose=1,
    mode="min",
    save_best_only=True,)

plot_image_callback = CustomCallback()

"""
Early Stopping Callback
Ensure we are not wasting resources
"""
early_stop_callback = keras.callbacks.EarlyStopping(
    monitor=early_stop_monitor, 
    min_delta=early_stop_min_delta,
    patience=early_stop_patience,
    verbose=1,
    mode="min",
    restore_best_weights=True,
    start_from_epoch=early_stop_start_epoch,
)

"""
CSV Logger Callback
"""
csv_callback = keras.callbacks.CSVLogger(filename=f"{folder_path}/csv_log_{formatted_time}.csv", separator=",", append=True)
