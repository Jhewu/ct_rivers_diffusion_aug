"""
THIS SCRIPT CONTAINS THE NECESSARY COMPONENTS 
TO BUILD THE U-NET BACKBONE FOR THE DIFFUSION MODEL
"""

""" ALL IMPORTS """
# Import necessary libraries
import tensorflow as tf
import keras
from keras import layers, initializers
import math

# Import from local scripts
from parameters import embedding_dims, attention_in_up_down_sample, attention_in_bottleneck, used_mix_precision

"""-------------------------------------------------CLASSES------------------------------------------------------"""
"""
Creating a custom layer called Sinusoidal Embedding
to replace Lambda layers that were causing  errors 
(probably due to dependencies reasons)
"""
@keras.saving.register_keras_serializable()
class SinusoidalEmbedding(layers.Layer): 
    def __init__(self, embedding_dims, **kwargs):
        super(SinusoidalEmbedding, self).__init__(**kwargs)
        self.embedding_dims = embedding_dims
    def build(self, input_shape):
        """
        Taken from the local sinusoidal_embedding 
        Build method is called when building the layer
        """
        self.embedding_min_frequency = 1.0
        self.embedding_max_frequency = 1000.0  # You can adjust this value
        frequencies = tf.exp(
            tf.linspace(
                tf.math.log(self.embedding_min_frequency),
                tf.math.log(self.embedding_max_frequency),
                self.embedding_dims // 2,
            )
        )
        angular_speeds = 2.0 * tf.constant(math.pi) * frequencies

        # Ensure the model works with mixed_precision as well
        if used_mix_precision: 
            datatype = tf.float16
        else: 
            datatype = tf.float32

        self.angular_speeds = tf.cast(angular_speeds, dtype=datatype)
        """
        We compute the frequencies for the sinusoidal embeddings 
        using exponential and logarithmic operations.
        """
    def call(self, x):
        """
        We compute the sinusoidal embeddings by concatenating sine
        and cosine functions of the angular speeds.
        The output embeddings contain both sine and cosine components.
        """
        embeddings = tf.concat(
            [tf.sin(self.angular_speeds * x), tf.cos(self.angular_speeds * x)], axis=-1
        )
        return embeddings

"""REVIEW THIS CODE TOMORROW"""
"""
Custom Attention Block to fix an error in Keras. 
For some reason, it only works if I wrap it on a Keras
layers, due to the difference between a Tf_fn and Keras layers
"""
@keras.saving.register_keras_serializable()
class AttentionBlock(layers.Layer):
    def __init__(self, num_heads=4, key_dim=128, dropout_rate=0.2, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Define the Reshape and MultiHeadAttention layers here
        channels = input_shape[-1]
        height = input_shape[1]
        width = input_shape[2]
        
        self.reshape1 = layers.Reshape((-1, channels))  # Flatten to (batch_size, sequence_length, channels)
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate
        )
        self.reshape2 = layers.Reshape((height, width, channels))  # Reshape back to original
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs):
        # Flatten the spatial dimensions
        x_reshaped = self.reshape1(inputs)
        
        # Apply multi-head attention
        attention_output = self.attention(x_reshaped, x_reshaped)  # Using x_reshaped as both query and value
        
        # Reshape back to the original spatial dimensions
        attention_output_reshaped = self.reshape2(attention_output)
        
        # Apply skip connection and dropout
        x = layers.Add()([inputs, attention_output_reshaped])
        x = self.dropout(x)
        
        return x
    
"""-------------------------------------------------U-NET BLOCKS------------------------------------------------------"""
"""
HIGH LEVEL SUMMARY: 
This function defines a residual block for a neural network
Residual blocks are layers which outputs are added to a deeper
layer
    - It checks the input width (input_width) of the tensor x.
    - If the input width matches the specified width, it sets the residual to be the same as x.
    - Otherwise, it applies a 1x1 convolution (layers.Conv2D) to transform the input tensor to the desired width.
    - Next, it applies batch normalization (layers.BatchNormalization) and two 3x3 convolutions with ReLU activation
        (swish activation is used here).
    - Finally, it adds the residual tensor to the output tensor and returns it.
"""
@keras.saving.register_keras_serializable()
def ResidualBlock(width): 
    # Width specify the number of output channels
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
             # Set residual to be the same as x if it matches
            residual = x
        else:
            # Set residual to the desired width
            residual = layers.Conv2D(width, kernel_size=1, kernel_initializer=initializers.HeNormal())(x) 
            x = layers.GroupNormalization(groups=8, axis=-1)(x)  
            x = layers.Activation(keras.activations.silu)(x)  
        
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=initializers.HeNormal())(x)
        x = layers.Activation(keras.activations.silu)(x)  
        x = layers.GroupNormalization(groups=8, axis=-1)(x)  
        
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=initializers.HeNormal())(x)
        x = layers.Activation(keras.activations.silu)(x)  
        x = layers.GroupNormalization(groups=8, axis=-1)(x)  
        
        x = layers.Add()([x, residual])
        return x
    return apply

"""
HIGH LEVEL SUMMARY: 
This function defines a downsampling block that reduces the spatial dimensions of the input tensor.

    - It expects a tuple (x, skips) as input, where x is the input tensor, and skips is a list to 
        store intermediate tensors.
    - It repeatedly applies block_depth residual blocks to the input tensor.
    - After that, it performs average pooling (reducing spatial dimensions) on the output tensor.
    - The function returns the downsampled tensor.
"""
@keras.saving.register_keras_serializable()
def DownBlock(width, block_depth): 
    # Width is number of output channels for the residual blocks
    # Block_depth determines how many residual blocks are stacked in this 
    # downsampling block
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
            # Add attention layer if specified
            if attention_in_up_down_sample: 
                x = AttentionBlock(num_heads=1, key_dim=width, dropout_rate=0)(x)
        x = layers.Conv2D(width, kernel_size=3, strides=2, padding="same", activation="swish", kernel_initializer=initializers.HeNormal())(x)
        return x
    return apply

"""
HIGH LEVEL SUMMARY:
This function defines an upsampling block that increases the spatial dimensions of the input tensor.
        - It also expects a tuple (x, skips) as input.
        - It first performs upsampling using bilinear interpolation (layers.UpSampling2D).
        - Then, it concatenates the upsampled tensor with the last tensor stored in skips.
        - It applies block_depth residual blocks to the concatenated tensor.
        - The function returns the upsampled tensor.
"""
@keras.saving.register_keras_serializable()
def UpBlock(width, block_depth):
    # same parameters as downblock with width and block_depth
    def apply(x):
        x, skips = x
        x = layers.Conv2DTranspose(width, kernel_size=3, strides=2, padding="same", activation="swish", kernel_initializer=initializers.HeNormal())(x)
        for _ in range(block_depth):
            a = skips.pop()
            x = layers.Concatenate()([x, a]) 
            x = ResidualBlock(width)(x)       
            # Add attention layer if specified
            if attention_in_up_down_sample: 
                x = AttentionBlock(num_heads=1, key_dim=width, dropout_rate=0)(x)
        return x
    return apply

"""-------------------------------------------------BUILDING THE UNET------------------------------------------------------"""
"""
HIGH LEVEL SUMMARY: 
    - Creates U-Net Model
        - The model takes inputs [noisy_images, noise_variances] and produces the denoised output.
        - The model is named “residual_unet”.
    - Uses the few functions mentioned above
    - Check comments for more information
"""
@keras.saving.register_keras_serializable()
def get_network(image_size, widths, block_depth):
    # Input for noisy images
    noisy_images = keras.Input(shape=(image_size[0], image_size[1], 3)) 
    
    # Input for noise variances
    noise_variances = keras.Input(shape=(1, 1, 1))                

    # Create a sinusoidal layer from our custom Sinusoidal embedding layer
    sinusoidal_layer = SinusoidalEmbedding(embedding_dims) 
    
    # Call the layer with your input (e.g., noise_variances)
    # Noise variances inputed to sinusoidal layer
    # then to upsampling using nearest neighbor interpolation
    e = sinusoidal_layer(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    # Noisy images input into Conv2D 
    # output is concatenated with e
    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = [] # skips stores the skip connections
    for width in widths[:-1]:
        # Series of downblocks are applied to the concatenated features
        # each downblock reduces spatial resolution and increases the number
        # of filters
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        # This is the bottleneck
        x = ResidualBlock(widths[-1])(x)
        
        # Add attention layer if specified
        if attention_in_bottleneck: 
            x = AttentionBlock(num_heads=4, key_dim=widths[-1]//8, dropout_rate=0.2)(x)
        
    for width in reversed(widths[:-1]):
        # Each block upsamples the features and reduces the number 
        # of filters
        x = UpBlock(width, block_depth)([x, skips])

    # Final convolution, 1x1 convolution with 3 channels (RGB) is applied to the output
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")

