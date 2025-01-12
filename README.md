    > You can either run a single model or run a training loop that trains
      3-6 diffusion models with the respective labels. If you want to run a single
      model, then 


    You can modify parameters

    You can either modify the parameters in config.py and run 
    'python3 run_diffusion.py' to run a single model, or if you want 
    to train 3-6 models in a training loop, 


    You can either modify parameters in config.py and run 'run_diffusion'
    to run single models, or if you want to train 3-6 models in a training loop, 
    use the argparse parameters below. 


    - --in_dir: 
    The directory containing the label folder with river images. 
    The structure should be similar to 'L1/1/image.JPG'.

    - --out_dir:
        The directory where model weights, callbacks, and results will be saved.

    - --runtime: 
        Specifies the run mode. Options are:
            - "training": for training the model.
            - "inference": for generating predictions using a trained model.
            - "inpainting": for using the model for image inpainting.

    - --load_and_train:
        A boolean flag indicating whether to load a pre-trained model for further training.

    - --eta:
        The eta parameter used for noise scheduling in the model's diffusion process.

    - --image_size: 
        A tuple specifying the size of the input images (height, width) for training or inference.

    - --num_epochs:
        The number of epochs for training the model.

    - --batch_size:
        The batch size used during training.

    - --learning_rate:
        The learning rate used for model training optimization.

    - --use_mix_precision:
        A boolean flag indicating whether to use mixed precision training to speed up training and reduce memory usage.

    - --gpu_index:
        The index of the GPU to be used for training.

    - --embedding_dims:
        The dimensions for embeddings used in the model.

    - --widths:
        A list of integers specifying the widths for each convolutional layer in the model's architecture.

    - --block_depth:
        The depth of the U-Net blocks used in the model.

    - --attention_in_bottleneck:
        A boolean flag indicating whether attention is used in the bottleneck layer of the U-Net.

    - --attention_in_up_down_sample:
        A boolean flag indicating whether attention is used in the up/down sampling layers of the U-Net.

    - --model_dir:
        The directory where the model's weight files are located.

    - --images_to_generate:
        The number of images to generate during inference or prediction.

    - --generate_diffusion_steps:
        The number of diffusion steps to take during image generation.