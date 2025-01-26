# Import the necessary libraries
import os
import gc
import argparse
import subprocess
import time

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

    # Parse arguments
    return parser.parse_args()

def RunDiffusion(args): 
    """
    This function runs the diffusion loop. It creates the parse arguments by extending 
    the list, and then pass it to run_diffusion.py
    """
    # Construct the command to run the script with arguments
    command = ['python3', 'run_diffusion.py']

    # General parameters
    if args.in_dir is not None:
        dataset_dir = args.in_dir
    else: raise IOError("Input directory (--in_dir) is required")
    if args.out_dir is not None:
        command.extend(['--out_dir', args.out_dir])

    # Training parameters
    if args.runtime is not None:
        command.extend(['--runtime', args.runtime])
    if args.load_and_train:
        command.extend(['--load_and_train'])
    if args.eta is not None:
        command.extend(['--eta', str(args.eta)])
    if args.image_size is not None:
        command.extend(['--image_size', str(args.image_size)])

    # Optimization (Training) parameters
    if args.num_epochs is not None:
        command.extend(['--num_epochs', str(args.num_epochs)])
    if args.batch_size is not None:
        command.extend(['--batch_size', str(args.batch_size)])
    if args.learning_rate is not None:
        command.extend(['--learning_rate', str(args.learning_rate)])
    if args.use_mix_precision:
        command.extend(['--use_mix_precision'])
    if args.gpu_index is not None:
        command.extend(['--gpu_index', str(args.gpu_index)])

    # U-Net architecture parameters
    if args.embedding_dims is not None:
        command.extend(['--embedding_dims', str(args.embedding_dims)])
    if args.widths is not None:
        command.extend(['--widths'])
        for width in args.widths: 
            command.extend([str(width)])  # Pass widths as a single string without commas
        # command.extend(['--widths', ' '.join(map(str, args.widths))])  # Assuming widths is a list
    if args.block_depth is not None:
        command.extend(['--block_depth', str(args.block_depth)])
    if args.attention_in_bottleneck:
        command.extend(['--attention_in_bottleneck'])
    if args.attention_in_up_down_sample:
        command.extend(['--attention_in_up_down_sample'])

    # Run the command using subprocess
    label_dirs = os.listdir(dataset_dir)

    for i in range(len(label_dirs)): 
        print(f"\nRunning diffusion model on label {label_dirs[i]}...\n")

        # Create the new directory for the label dataset
        label_dataset_path = os.path.join(dataset_dir, label_dirs[i])

        # Extend the new directory to the command list
        command.extend(['--in_dir', label_dataset_path])

        try: 
            # Run the subprocess  
            result = subprocess.run(command, check=True, 
                                    capture_output=True,
                                    text=True)

            # Print the result
            print("Output:", result.stdout)  # Output from the script
            print("Error:", result.stderr)   # Any errors that occurred
            
            # Attempt at clearing memory
            del result
            gc.collect()
            
            # Wait for a bit
            time_to_wait = 5
            print(f"Waiting for {time_to_wait} seconds, until the next execution...")
            time.sleep(time_to_wait)
            
        except subprocess.CalledProcessError as e: 
            # Handle the error and print more detailed information
            print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
            print("Output:", e.output)
            print("Error:", e.stderr)
            

    
if __name__ == "__main__":
    des=""" 
    ------------------------------------------
    - Diffusion Training Loop (Overview) - 

    This script serves as a training loop for run_diffusion.py. 
    It uses argparse to parse all of the arguments requested to run
    run_diffusion.py. NOTE: to run run_diffusion.py, you do not require
    to pass any arguments, as the config.py file has default values. 

    However, run_training_loop, will require you to provide "--in_dir", 
    which is the location of your dataset. If you use my diffusion 
    augmentation script, the in_dir path is just: "pwd/diffusion_data."
    
    ------------------------------------------
    - Changing Parameters for the Diffusion Model -

    There are two ways to change parameters: 

    (1) Directly modify the config.py
    (2) Use argparse to pass arguments. NOTE: not all parameters are
        available for argparse, if you wish to modify them, modify the 
        config.py directly. 

    NOTE: If you're using run_training_loop.py, the config.py file copied
    to the output dir will not reflect the parameters you used (because
    it does not update from argparse), but rather you should focus on the 
    config_parameters.txt file

    ------------------------------------------
    - How to Use -

    > If you want to use default parameters: 

    'python3 run_training_loop.py --in_dir pwd/diffusion_data

    > If you want to change some parameters: 

    'python3 run_training_loop.py --in_dir pwd/diffusion_data
    --learning_rate 1e-5 --batch_size 8

    ------------------------------------------
    """    
 
    # Parse all of the arguments
    args = ParseArgs(des)

    # Run the diffusion model subprocess loop
    RunDiffusion(args) 

    print("\nFinish training, please check the results directory\n")
