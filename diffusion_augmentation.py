"""
Please review the description located at the 
bottom of the script for more information
"""

# Import external libraries
import os
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
import gc
import time

# Import script to check if data is balanced
from utils import *

DATASET_NAME = "toy_dataset"                    # The directory where the labels are stored in
DEST_FOLDER = "balanced_data"                   
WEIGHT_PATHS = []
DIFFUSION_STEPS = 50

def DiffusionAugmentation(images_to_aumgment, dest_dir, weight_path, diffusion_steps): 
    """
    Balances a label folder by starting with the site id
    with the least amount of images until 'images_to_augmnet'
    is met
    """
    # Create the commands
    command = ["python3", "run_diffusion.py", 
               "--runtime", "inference",
               "--out_dir", dest_dir, 
               "--model_dir", weight_path, 
               "--generate_diffusion_steps", str(diffusion_steps), 
               "--images_to_generate", str(images_to_aumgment), 
               "--subprocess"]

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
    print(f"\nWaiting for {time_to_wait} seconds, until the next execution...\n")
    time.sleep(time_to_wait)

def LabelBalancer(input_dir, output_dir, weight_paths, diffusion_steps):
    """
    This function will determine which label images to 
    augment based on the number of images in the two 
    category problem (label 1,2,3) and (label, 4, 5, 6)
    """
    # Obtain image directory
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, input_dir)
    dest_dir = os.path.join(root_dir, output_dir)

    # Initialize the lists
    category_1_images = [] 
    category_2_images = []

    # The line below basically does this: 
    #                       os.path.join(dataset_dir, "1"),
    #                       os.path.join(dataset_dir, "2"), 
    #                       os.path.join(dataset_dir, "3"), 
    #                       os.path.join(dataset_dir, "4"), 
    #                       os.path.join(dataset_dir, "5"), 
    #                       os.path.join(dataset_dir, "6")]   
    all_label_dirs = [os.path.join(dataset_dir, str(i)) for i in range(1, 7)]

    # Prepare category 1
    category_1_list = [
        os.listdir(all_label_dirs[0]), 
        os.listdir(all_label_dirs[1]), 
        os.listdir(all_label_dirs[2])]
    
    for i in range(len(category_1_list)): 
        category_1_images.extend(category_1_list[i])
    
    category_1_count = len(category_1_images)

    # Prepare category 2
    category_2_list = [
        os.listdir(all_label_dirs[3]), 
        os.listdir(all_label_dirs[4]), 
        os.listdir(all_label_dirs[5])]

    for i in range(len(category_2_list)): 
        category_2_images.extend(category_2_list[i])
    
    category_2_count = len(category_2_images)

    # Create the destination directories
    # The line below is similar to all_label_dirs
    dest_label_dirs = [os.path.join(dest_dir, str(i)) for i in range(1, 7)]

    # Case #1 
    if category_1_count > category_2_count: 
        print("\nCategory 1 larger than Category 2\n")

        images_to_augment = category_1_count - category_2_count
        print(f"Images to augment: {images_to_augment}\n")

        # Copy all the original files to the destination directory
        print(f"Copying original files to {dest_dir}...\n")
        max_workers = 10
        with ThreadPoolExecutor(max_workers=max_workers) as executor: 
            executor.map(Copy_dir, all_label_dirs, dest_label_dirs)

        # Get counts of each label in category 2
        # Similar to category_2_count, but instead a list of lengths
        category_2_count_split = [
            len(category_2_list[0]), 
            len(category_2_list[1]), 
            len(category_2_list[2])]
        
        # Compute the complementary probabilities for sampling
        p_label_1 = 1 - (category_2_count_split[0] / category_2_count)
        p_label_2 = 1 - (category_2_count_split[1] / category_2_count)
        p_label_3 = 1 - (category_2_count_split[2] / category_2_count)

        # Normalize the probabilities to 1, before sampling
        p_total = p_label_1 + p_label_2 + p_label_3
        p_label_1 /= p_total
        p_label_2 /= p_total
        p_label_3 /= p_total

        # Use numpy to sample which labels to augment
        sample = list(np.random.choice([1, 2, 3], images_to_augment, p=[p_label_1, p_label_2, p_label_3], replace=True))
        
        # Count how many images to augment for each label
        images_to_aug_per_label = [sample.count(1), sample.count(2), sample.count(3)]

        # Comment this out, if you want to double-check if the number
        # of images are the same
        print("\nLabel 4,5,6 should contain the respective number of images. If not, the balancing was incorrect\n")
        print(images_to_aug_per_label[0] + category_1_count_split[0])
        print(images_to_aug_per_label[1] + category_1_count_split[1])
        print(images_to_aug_per_label[2] + category_1_count_split[2])
        print()

        # Perform the data augmentation (reverse diffusion for each model)
        for i in range(len(images_to_aug_per_label)):
            print(f"Augmenting label {i+3}...\n")
            DiffusionAugmentation(images_to_aug_per_label[i], 
                                dest_label_dirs[i+3], 
                                weight_paths[i], 
                                diffusion_steps)
            
    # Case #2
    else: 
        print("\nCategory 2 larger than Category 1\n")

        images_to_augment = category_2_count - category_1_count
        print(f"Images to augment: {images_to_augment}\n")

        # Copy all the original files to the destination directory
        print(f"Copying original files to {dest_dir}...\n")
        max_workers = 10
        with ThreadPoolExecutor(max_workers=max_workers) as executor: 
            executor.map(Copy_dir, all_label_dirs, dest_label_dirs)

        # Get counts of each label in category 1
        # Similar to category_1_count, but instead a list of lengths
        category_1_count_split = [
            len(category_1_list[0]), 
            len(category_1_list[1]), 
            len(category_1_list[2])]
        
        # Compute the complementary probabilities for sampling
        p_label_1 = 1 - (category_1_count_split[0] / category_1_count)
        p_label_2 = 1 - (category_1_count_split[1] / category_1_count)
        p_label_3 = 1 - (category_1_count_split[2] / category_1_count)

        # Normalize the probabilities to 1, before sampling
        p_total = p_label_1 + p_label_2 + p_label_3
        p_label_1 /= p_total
        p_label_2 /= p_total
        p_label_3 /= p_total

        # Use numpy to sample which labels to augment
        sample = list(np.random.choice([1, 2, 3], images_to_augment, p=[p_label_1, p_label_2, p_label_3], replace=True))
        
        # Count how many images to augment for each label
        images_to_aug_per_label = [sample.count(1), sample.count(2), sample.count(3)]

        # Comment this out, if you want to double-check if the number
        # of images are the same
        print("\nLabel 1,2,3 should contain the respective number of images. If not, the balancing was incorrect\n")
        print(images_to_aug_per_label[0] + category_1_count_split[0])
        print(images_to_aug_per_label[1] + category_1_count_split[1])
        print(images_to_aug_per_label[2] + category_1_count_split[2])
        print()

        # Perform the data augmentation (reverse diffusion for each model)
        for i in range(len(images_to_aug_per_label)):
            print(f"Augmenting label {i+1}...\n")
            DiffusionAugmentation(images_to_aug_per_label[i],
                                  dest_label_dirs[i],
                                  weight_paths[i], 
                                  diffusion_steps)

if __name__ == "__main__":
    des="""
    ------------------------------------------
    - CT DEEP Rivers Diffusion Augmentation for Image Classification (Overview) -

    Balances and perform the necessary image augmentation
    (e.g. reverse diffusions from a pre-trained diffusion model)
    to rivers stream images from CT DEEP, to prepare the dataset
    for image classification training. This iteration focuses on
    the two-category problem (label 1,2,3 and label 4,5,6).
    ------------------------------------------
    - The Augmentation Process -

    The script will augment the category with the least amount of images. 
    After determining which category has less images, it will augment more
    the labels with lesser amount of images, and within each label, it will 
    augment more the site_ids with the least amount of images. 
    ------------------------------------------
    - How to Use -

    > in_dir (required): directory containing the labeled folders (e.g. 1,2...6)
    > weight_paths (required):
    > out_dir (optional): default is "balanced_data" in current working directory
    > generate_diffusion_steps (optional): default is 50
    
    ------------------------------------------
    """
    # Initialize the Parser
    parser = argparse.ArgumentParser(description=des.lstrip(" "),formatter_class=argparse.RawTextHelpFormatter)

    # Add the arguments
    parser.add_argument('--in_dir',type=str,help='input directory of images with labeled subfolders')
    parser.add_argument('--weight_paths', type=str, nargs='+', help="paths of the weights")
    parser.add_argument('--out_dir',type=str,help='output directory prefix')
    parser.add_argument('--generate_diffusion_steps', type=int, help="number of diffusion steps for generation")

    # Parse the arguments
    args = parser.parse_args()

    # Assign the values
    if args.in_dir is not None:
        DATASET_NAME = args.in_dir
    else: raise IOError
    if args.weight_paths is not None:
        WEIGHT_PATHS = args.weight_paths
    else: raise IOError
    if args.out_dir is not None:
        DEST_FOLDER = os.path.join(args.out_dir, "balanced_data")
    else: DEST_FOLDER
    if args.generate_diffusion_steps is not None:
        DIFFUSION_STEPS = args.generate_diffusion_steps
    else: DIFFUSION_STEPS

    params = {'in_dir':DATASET_NAME,'out_dir':DEST_FOLDER,
              'weights_path':WEIGHT_PATHS, "diffusion_steps": DIFFUSION_STEPS}
    print('\nUsing params:%s'%params)

    # Call the function
    LabelBalancer(DATASET_NAME, DEST_FOLDER, WEIGHT_PATHS, DIFFUSION_STEPS)
    print(f"\nFinish balancing the labels\nCheck your directory for {DEST_FOLDER}")

