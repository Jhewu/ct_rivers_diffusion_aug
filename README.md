# CT Rivers Diffusion Model

## Overview

This research project aims to improve image classification accuracy for CT river images by using diffusion models to artificially augment the dataset. This approach ensures a balanced dataset, which is often it's not the case in environmental image datasets, and exposes the classification model to more image diversity, leading to better generalization.

![Generated Sample #1](./sample_images/2025-01-08_23_37_11_generated_img_1.jpg)

This repository contains 3 main components: 
1. `site_id_balancer.py`, which augments the original dataset with traditional augmentations (e.g., rotations and horizontal flips). This new dataset named `diffusion data` will be used to train the diffusion model to ensure no site_id bias during inference
2. The diffusion model itself. The model scripts includes `run_diffusion.py`, `diffusion_model.py`, `u_net_backbone.py`, `config.py`, `utils.py`, and `train_all.py.` This is the main script used to train the model, and perform inference. 
3. `diffusion_augmentation.py` which uses pre-trained diffusion models (after `train_all.py`) to augment the final dataset to be used to train the image classifier

This repository is designed to be used in that respective order.

## Site ID Balancer (for Diffusion Models)

This script balances and performs the necessary image augmentations (e.g., 5-degree rotations + horizontal flips) on river stream images from CT DEEP, to prepare the dataset for diffusion model training. Since the dataset is unbalanced, training it directly (without augmentation) may cause the diffusion model to generate more images from certain site IDs. To compensate for this, we augment the dataset, ensuring a balanced distribution for training.

### The Augmentation Process

The default augmentation process includes:
- **Rotations**: Every 5 degrees, from 5 to 30 degrees.
- **Horizontal flip**: Applied before moving to the next degree of rotation.

The script augments all images from each `site_id` folder until the number of images matches the folder with the maximum number of images. If there are not enough unique augmentations to match the maximum number of images, the process will stop after running out of available augmentations. This ensures that all `site_ids` are adequately represented in the dataset.

### How to Use

The script takes several input arguments:

- **in_dir**: (required) The directory containing labeled folders (e.g., `1`, `2`, ... `6`).
- **out_dir** (optional): The directory where augmented images will be saved. If not provided, the augmented images will be saved in the current working directory. The dest directory will be structured like this and it's normal: "balanced_data_for_diffusion/flow_1/1" and so on. This is done so to make it easier to fetch when training it on the diffusion model.
- **labels** (default = 3): The number of labels (site IDs) to balance and augment. For example:
  - If `labels=3`, the script will augment labels `1`, `2`, and `3`.
  - If `labels=6`, the script will augment labels `1` through `6`.
- **theta** (default = 5): The angle of rotation in degrees. If you change `theta`, you should also adjust the `fact` parameter to ensure the image is properly cropped after rotation. The default setting of 5-degree intervals requires a `fact` of 1.3 to account for the padding caused by rotation.
- **fact** (default = 1.3): The zoom factor applied after rotation to crop out any black padding that may result from the image rotation.
- **multiplier** (default = 6): The number of rotations to apply. If `multiplier=6`, the image will be rotated from 5 to 30 degrees, with horizontal flips in between, resulting in a final multiplier of 13x for each image.

#### Example Usage
```bash
python3 site_id_balancer.py --in_dir /path/to/images --labels 3 --theta 5 --fact 1.3 --multiplier 6
```

## Diffusion Models 

The model is a DDIM Diffusion model (with a 50% stochastic process) built on a U-Net backbone, implemented in Tensorflow/Keras.

### Changing Parameters for the Diffusion Model

There are two ways to change the parameters for the diffusion model:

1. **Modify `config.py` directly:**  
   You can open the `config.py` file and manually modify the parameters. This provides full control over all available settings.
   
2. **Use `argparse` to pass arguments:**  
   You can also pass parameters via the command line using argparse. To keep this `README.md` file concise, please refer to `utils.py` for the full lists of argparse. **Note:** Not all parameters are available via argparse. If you need to modify parameters not exposed through argparse, you'll need to update the `config.py` directly.

### How to Use (Training & Inference)

To train a single model or perform inference on a single model, use `run_diffusion.py` only. 

#### Training the Model (using `config.py`)

1. Set `runtime = "training"` in the `config.py`.
2. Set `in_dir` to the dataset directory, including the label (e.g., `diffusion_data/L2`).
3. Modify other important training parameters like `num_epochs`, `batch_size`, or `learning_rate`.
4. Run the following command in the terminal:  

    ```
    python3 run_diffusion.py 
    ```

#### Training the Model (using `argparse`)

1. Add as many `argparse` arguments like below:

    ```
    python3 run_diffusion.py --in_dir diffusion_data/L2 --learning_rate 1e-5 --batch_size 8
    ```

#### Perform Inference (using `config.py`):

1. Train a model, and ensure `config.py` is the same as during training
2. Modify `model_dir` to the folder where the weight is located, e.g., `"results/L2_2025-01-12_14:55:51"`.
3. Modify other important parameters such as `images_to_generate` and `generate_diffusion_steps`, and then run the following command: 

    ```
    python3 run_diffusion.py
    ```

#### Perform Inference (using `argparse`):

1. Set `--runtime inference` and provide the directory of the model's weights `--model_dir` like this:   

    ```
    python3 run_diffusion --runtime inference --model_dir results/L2_2025-01-12_14:55:51
    ```

### Training Multiple Models in a Loop

To train multiple models, use `train_all.py`. `train_all.py` serves as a training loop for `run_diffusion.py`. It uses argparse to parse all of the arguments requested to run `run_diffusion.py`. **NOTE:** to run `run_diffusion.py`, you do not need to pass any arguments, as the `config.py` file contains default values.

However, `train_all.py` will require you to provide `--in_dir`, which is the location of your dataset. If you used `site_id_balancer.py` then the `in_dir` path is simply: `pwd/diffusion_data`.

#### Changing Parameters Within the Training Loop

There are two ways to change parameters:

1. Directly modify the `config.py`.
2. Use argparse to pass arguments. The `train_all.py` contains the same argparse as `run_diffusion.py` **NOTE:** Not all parameters are available for argparse. If you wish to modify them, edit the `config.py` directly.

**NOTE:** If you're using `train_all.py`, the `config.py` file copied to the output directory will not reflect the parameters you used (because it does not update from argparse). Instead, focus on the `config_parameters.txt` file.

#### How to Use `run_training_loop.py`

If you want to use default parameters:  

```
python3 run_training_loop.py --in_dir pwd/diffusion_data
```

If you want to change some parameters: 

```
python3 run_training_loop.py --in_dir pwd/diffusion_data --learning_rate 1e-5 --batch_size 8
```

## Diffusion Augmentation 


