"""
THIS CODE IS THE MAIN CLASS BUILDER FOR
OUR DIFFUSION MODEL.
"""

# Import necessary libraries
import tensorflow as tf
import keras
from keras import layers
from keras import ops
import numpy as np

# Import from local scripts
from u_net_backbone import get_network

@keras.saving.register_keras_serializable()
class DiffusionModel(keras.Model): 
    def __init__(self, image_size, widths, block_depth, eta, 
                 max_signal_rate, min_signal_rate, batch_size, 
                 ema, embedding_dims, attention_in_up_down_sample, attention_in_bottleneck):
        super().__init__()

        self.normalizer = layers.Normalization()                    # for pixel normalization
        self.network = get_network(image_size, widths, block_depth, 
                                   embedding_dims, attention_in_up_down_sample, 
                                   attention_in_bottleneck) # obtaining the U-NET    
        self.image_size = image_size
        self.ema_network = keras.models.clone_model(self.network)   # EMA version of the network
        self.eta = eta                                              # the amount of stochastic noise added back
        self.max_signal_rate = max_signal_rate
        self.min_signal_rate = min_signal_rate
        self.batch_size = batch_size
        self.ema = ema

    def compile(self, **kwargs):
        """
        Compile method is overridden to create custom metrics
        such as noise loss, and image loss
        these metrics will be tracked during training. 
        """
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss") # initializing the metrics

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker] 

    def denormalize(self, images):
        """
        Convert the pixel values back to 0-1 range
        """
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return ops.clip(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        """
        Determines how many steps to sample/denoise
        The number of steps is implicit. This is a 
        simplified, continuos version of the cosine 
        schedule, commonly used in the literature

        Please refer to the Keras DDIM Tutorial for more 
        information: 
        https://keras.io/examples/generative/ddim/#hyperparameters 
        """        
        # Convert diffusion times to angles
        start_angle = ops.cast(ops.arccos(self.max_signal_rate), "float32")
        end_angle = ops.cast(ops.arccos(self.min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # Use angles to calculate signal and noise rates
        signal_rates = ops.cos(diffusion_angles)
        noise_rates = ops.sin(diffusion_angles)
        # Note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        """
        Uses the network (either main or EMA) to 
        predict noise components and calculate image components 
        """
        # The exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # Predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """
        Custom implementation of both DDIM and DDPM sampling 
        (reverse diffusion) process. If use DDIM sampling, set eta=0, 
        if use DDPM sampling, set eta=1, or use hybrid in between such 
        as eta=0.7
        """
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise
    
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
    
            # Calculate current timestep t and next timestep t-1
            current_time = ops.ones((num_images, 1, 1, 1)) - step * step_size
            # next_time = current_time - step_size
            next_time = tf.clip_by_value(current_time - step_size, 0.0, 1.0)
            
            # Get noise and signal rates for both timesteps
            noise_rates, signal_rates = self.diffusion_schedule(current_time)
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_time)
    
            # Predict noise and clean image components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False)
    
            # Add stochastic noise component (DDPM)
            if self.eta > 0:                
                # Scale random noise by both eta and noise rate
                noise = keras.random.normal(shape=noisy_images.shape)
                    
                # Compute variance with numerical stability
                alpha_t = ops.maximum(tf.square(signal_rates), 1e-7)
                alpha_s = ops.maximum(tf.square(next_signal_rates), 1e-7)
                
                variance = ops.clip(
                    (1.0 - alpha_s/alpha_t) * (1.0 - alpha_t) / (1.0 - alpha_s),
                    0.0, 1.0
                )
                variance = self.eta * variance
                noise_scale = tf.sqrt(variance)
                stochastic_term = noise_scale * noise

                # Combine deterministic and stochastic components
                next_noisy_images = (
                    next_signal_rates * pred_images + 
                    next_noise_rates * pred_noises +
                    stochastic_term)
            else: 
                # Combine predicted components for next step
                next_noisy_images = (
                    next_signal_rates * pred_images + next_noise_rates * pred_noises)

        return pred_images
    
    def reverse_diffusion_single(self, initial_noise, diffusion_steps):
        """
        Custom Reverse Diffusion (DDPM Sampling schedule). If use DDIM sampling 
        schedule, set self.eta = 0
        - Performs reverse diffusion (sampling), not simultaneously, but per image
        - The processing time will be longer, but it requires less GPU ram
        """
        step_size = 1.0 / diffusion_steps
        pred_images = []

        for i in range(initial_noise.shape[0]):  # Iterate over each image
            next_noisy_image = initial_noise[i]
            
            for step in range(diffusion_steps):
                noisy_image = next_noisy_image

                # Calculate current timestep t and next timestep t-1
                current_time = ops.ones((1, 1, 1, 1)) - step * step_size
                next_time = current_time - step_size

                # Get noise and signal rates for both timesteps
                noise_rates, signal_rates = self.diffusion_schedule(current_time)
                next_noise_rates, next_signal_rates = self.diffusion_schedule(next_time)

                # Predict noise and clean image components
                pred_noise, pred_image = self.denoise(
                    noisy_image[None, ...], noise_rates, signal_rates, training=False)
                    # Network used in eval mode
            
                # Add stochastic noise component (DDPM)
                if self.eta > 0:
                    # Scale random noise by both eta and noise rate
                    noise = keras.random.normal(shape=noisy_image.shape)
                        
                    # Compute variance for stochastic term using cosine schedule
                    alpha_t = ops.maximum(tf.square(signal_rates), 1e-7)
                    alpha_s = ops.maximum(tf.square(next_signal_rates), 1e-7)

                    variance = ops.clip(
                        (1.0 - alpha_s/alpha_t) * (1.0 - alpha_t) / (1.0 - alpha_s),
                        0.0, 1.0
                    )
                    variance = self.eta * variance
                    noise_scale = tf.sqrt(variance)
                    stochastic_term = noise_scale * noise
    
                    # Combine deterministic and stochastic components
                    next_noisy_image = (
                        next_signal_rates * pred_image + 
                        next_noise_rates * pred_noise +
                        stochastic_term)[0]
                else:
                    # Combine predicted components for next step
                    next_noisy_image = (
                        next_signal_rates * pred_image + next_noise_rates * pred_noise)[0]
                        # This new noisy image will be used in the next step
            pred_images.append(pred_image[0])
        return np.stack(pred_images)
        

    def generate(self, num_images, diffusion_steps, single):
        """
        Main function used to generated images, used in both 
        train/test and also for inference. Single = False decides
        if to generate the image by batches, if done by batches all 
        generated images look similar, and increase GPU RAM
        usage, because it's doing parallel processing. I recommend setting
        single = True, when inference
        """
        initial_noise = keras.random.normal(
            shape=(num_images, self.image_size[0], self.image_size[1], 3)
        )
        if single == True:
            generated_images = self.reverse_diffusion_single(initial_noise, diffusion_steps)
            generated_images = self.denormalize(generated_images)
        else:
            generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
            generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        """
        Main training step, losses and backpropagation is performed here
        """
        # Normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True) 
        noises = keras.random.normal(shape=(self.batch_size, self.image_size[0], self.image_size[1], 3))

        # Sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        
        # Calculate noise rates and signal rates based on diffusion times.
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        
        # Mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # Train the network to separate noisy images to their components
            # denoise images into predicted noise and image components
            pred_noises, pred_images = self.denoise( 
                noisy_images, noise_rates, signal_rates, training=True
            )

            # Compute noise loss (used for training) and image loss (used as a metric).
            noise_loss = self.loss(noises, pred_noises)  
            image_loss = self.loss(images, pred_images)  
            """
            tf.GradientTape() Usage:
            - You create a tf.GradientTape context.
            - Inside this context, you perform operations (e.g., forward pass, loss computation) 
              involving TensorFlow variables (usually tf.Variables).
            - The tape records these operations.
            - When you exit the context, you can compute gradients with respect to the recorded 
              operations using the tape.
            """
        # Compute gradients and update network weights using the optimizer.
        gradients = tape.gradient(noise_loss, self.network.trainable_weights)

        # The zip(...) function pairs up corresponding elements from gradients and trainable_weights
        # each pair consists of a gradient and the corresponding trainable weight. 
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # Update the metrics
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # Track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
                # Update exponential moving averages of network weights.

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        """
        Main testing step, similar to the training step, but without backpropagation
        """
        # Normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = keras.random.normal(shape=(self.batch_size, self.image_size[0], self.image_size[1], 3))

        # Sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        # Use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        # Calculate the loss
        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        # Update the loss
        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}

    def repaint(self, img, mask, diffusion_steps):
        """
        Custom implementation of RePaint inpainting from the paper: 
        "RePaint: Inpainting using Denoising Diffusion Probabilistic Models"
        https://arxiv.org/abs/2201.09865 

        RePaint-style inpainting with context preservation and noise resampling
        Args:
            img: Input image to be inpainted
            mask: Binary mask where 1 indicates regions to keep (known regions)
            diffusion_steps: Number of diffusion steps
        """
        
        # Normalize masks and image
        norm_mask = tf.cast(mask/255.0, dtype=tf.float32)  # Areas to preserve
        norm_inverse_mask = tf.cast((255 - mask)/255.0, dtype=tf.float32)  # Areas to inpaint
        norm_img = tf.cast(self.normalizer(img/255.0)[0], dtype=tf.float32)

        # Initialize noise
        initial_noise = keras.random.normal(
            shape=(self.image_size[0], self.image_size[1], 3)
        )

        """MODIFY LATER TO PERFORM THIS IN BATCHES"""
        # Calculate step size for diffusion
        step_size = 1.0 / diffusion_steps

        # Current_sample is the unknown 
        current_sample = initial_noise

        for step in range(diffusion_steps):
            """Denoising Process"""
            # Calculate current timestep and rates
            diffusion_time = ops.ones((1, 1, 1, 1)) - step * step_size
            noise_rate, signal_rate = self.diffusion_schedule(diffusion_time)

            # Create the combined image with context preservation
            # For known regions (norm_mask), use the original image
            # For unknown regions (norm_inverse_mask), use the current noisy sample
            combined_img = (
                norm_mask * norm_img +              # Keep original content in known regions
                norm_inverse_mask * current_sample  # Use current sample in regions to inpaint
            )

            # Predict noise and denoised image
            pred_noise, pred_image = self.denoise(
                combined_img[None, ...], 
                noise_rate, 
                signal_rate, 
                training=False
            )

            """Forward Process"""
            # Calculate next timestep rates
            next_t = diffusion_time - step_size
            next_noise_rate, next_signal_rate = self.diffusion_schedule(next_t)

            # Mix predicted components for next step
            next_sample = (
                next_signal_rate * pred_image + 
                next_noise_rate * pred_noise
            )[0]

            # Apply noise resampling in masked regions (RePaint key feature)
            # Only resample noise in early and middle steps, not near the end
            """REVIEW THIS PART OF THE CODE"""
            if step < int(diffusion_steps * 0.2):  # Don't resample in final 20% of steps
                new_noise = keras.random.normal(next_sample.shape)
                next_sample = (
                    norm_inverse_mask * new_noise +  # Resample noise in regions to inpaint
                    norm_mask * next_sample  # Keep current sample in known regions
                )

            current_sample = next_sample
            """ALSO REVIEW THIS PART OF THE CODE"""
            # Optional: periodically preserve known regions to prevent drift
            if step % 10 == 0:  # Every 10 steps
                current_sample = (
                    norm_mask * norm_img +
                    norm_inverse_mask * current_sample
                )

        current_sample = self.denormalize(current_sample)

        # Return final result
        return current_sample
