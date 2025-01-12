# CT Rivers Diffusion Model

## Overview

This project aims to improve image classification accuracy for CT river images by using diffusion models to artificially augment the dataset. This approach ensures a balanced dataset, which is often not the case in environmental image datasets, and exposes the classification model to more image diversity, leading to better generalization.

## Changing Parameters for the Diffusion Model

There are two ways to change the parameters for the diffusion model:

1. **Modify `config.py` directly:**  
   You can open the `config.py` file and manually modify the parameters. This provides full control over all available settings.
   
2. **Use `argparse` to pass arguments:**  
   You can also pass parameters via the command line using argparse. **Note:** Not all parameters are available via argparse. If you need to modify parameters not exposed through argparse, you'll need to update the `config.py` directly.

## How to Use (Training & Inference)

To train a single model or perform inference on a single model, use `run_diffusion.py` only. 

### Training the Model (using `config.py`)

1. Set `runtime = "training"` in the `config.py`.
2. Set `in_dir` to the dataset directory, including the label (e.g., `diffusion_data/L2`).
3. Modify other important training parameters like `num_epochs`, `batch_size`, or `learning_rate`.
4. Run the following command in the terminal:  

    ```
    python3 run_diffusion.py 
    ```

### Training the Model (using `argparse`)

1. Add as many `argparse` arguments like below:

    ```
    python3 run_diffusion.py --in_dir diffusion_data/L2 --learning_rate 1e-5 --batch_size 8
    ```

### Perform Inference (using `config.py`):

1. Train a model, and ensure `config.py` is the same as during training
2. Modify `model_dir` to the folder where the weight is located, e.g., `"results/L2_2025-01-12_14:55:51"`.
3. Modify other important parameters such as `images_to_generate` and `generate_diffusion_steps`, and then run the following command: 

    ```
    python3 run_diffusion.py
    ```

### Perform Inference (using `argparse`):

1. Set `--runtime inference` and provide the directory of the model's weights `--model_dir` like this:   

    ```
    python3 run_diffusion --runtime inference --model_dir results/L2_2025-01-12_14:55:51
    ```

## Training multiple models

To train multiple models, use `run_training_loop.py`. `run_training_loop.py` serves as a training loop for `run_diffusion.py`. It uses argparse to parse all of the arguments requested to run `run_diffusion.py`. **NOTE:** to run `run_diffusion.py`, you do not need to pass any arguments, as the `config.py` file contains default values.

However, `run_training_loop` will require you to provide `--in_dir`, which is the location of your dataset. If you use my diffusion augmentation script, the `in_dir` path is simply: `pwd/diffusion_data`.

### Changing Parameters for Within the Training Loop

There are two ways to change parameters:

1. Directly modify the `config.py`.
2. Use argparse to pass arguments. **NOTE:** Not all parameters are available for argparse. If you wish to modify them, edit the `config.py` directly.

**NOTE:** If you're using `run_training_loop.py`, the `config.py` file copied to the output directory will not reflect the parameters you used (because it does not update from argparse). Instead, focus on the `config_parameters.txt` file.

### How to Use `run_training_loop.py`

#### If you want to use default parameters:  

```
python3 run_training_loop.py --in_dir pwd/diffusion_data
```

#### If you want to change some parameters: 

```
python3 run_training_loop.py --in_dir pwd/diffusion_data --learning_rate 1e-5 --batch_size 8
```