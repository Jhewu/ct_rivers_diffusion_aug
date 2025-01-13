"""
TO VIEW THIS SCRIPT'S FUNCTION
SCROLL DOWN TO IF__MAIN__
"""

# Import external libraries
import os
import argparse
import random
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import helper functions
from utils import *

"""GLOBAL PARAMETERS"""
THETA = 5                                    
FACT = 1.3                                      
MULTIPLIER = 6                                
DATASET_DIR = "flow_600_200"
DEST_DIR = "diffusion_data"
LABEL = 3
SEED = 42

"""------Main Runtime------"""
def DataBalancer(input_dir, dest_dir, theta, fact, label, seed, multiplier):
    # Obtain image directory
    image_list = os.listdir(input_dir)

    # Declare dictionary to count site_id
    site_ids_count = {}

    # Declare dictionary to store the full_paths of
    # each image within a site id
    site_path = {}      # dictionary with a list inside
                        # e.g. [site_id: 23]

    # Count site_id and append full_file_path
    total_files = 0
    for file_name in image_list:
        total_files += 1
        
        # Obtain full file path
        full_file_path = os.path.join(input_dir, file_name)

        # Obtain the site_id
        site_id = file_name.split("_")[0]
    
        if site_id in site_ids_count:
            # Case 1: if it's already in the list
            site_ids_count[site_id] += 1
            site_path[site_id].append(full_file_path)
        else:
            # Case 2: if it's not in the list
            site_ids_count[site_id] = 1
            site_path[site_id] = [full_file_path]

    # Sort the dictionary, and turn it into a list
    sorted_list = SortDict(site_ids_count)
    max = sorted_list[-1][1]

    # Keep track of this number to ensure, the augmentation
    # is working correctly
    total_augmented_images = 0

    # This loop is iterating through each site_id
    for element in sorted_list: 
        # Element is shaped like this: ('site_id', 2)
        # first element is the site_id (a string)
        # second element is the count

        # Defined the site id
        site = element[0]

        # Randomly shuffle the list contents
        # so that the balancing is not deterministic
        random.seed(seed)
        random.shuffle(site_path[site])

        # Now augment the original images,
        # to reach the max. The code is applying
        # the minimum degree of rotation to each
        # image in order to reach the image count
        images_to_augment = max - site_ids_count[site]          # --> number of images to augment to reach max
        maximum_augmentation = site_ids_count[site]*(multiplier*2)
        counter = 1                                             # --> counter for number of images currently augmentated
        switch = -1                                             # --> (see more down), rotations only or horizontal flipped rotations
        theta_local = theta
        fact_local = fact

        # After the switch is triggered two times, 
        # increase THETA and FACT            
        switch_ended = 1    
        while counter <= maximum_augmentation and counter <= images_to_augment: # --> continue to augment in case rotation or flip rotation is not enough
            # print(f"There are {images_to_augment} images to augment in site {site}, and we are in {counter}")

            # This loop is iterating each image path
            for i in range(len(site_path[site])):
                # Get image path
                image_path = site_path[site][i]

                # Obtain the image name
                basename = os.path.basename(image_path)
                image_name = basename.split(".JPG")[0].split(".jpg")[0]
                new_destination = os.path.join(dest_dir, f"{image_name}_{theta_local}_{switch}.JPG")

                if counter > maximum_augmentation or counter > images_to_augment:       # --> if we reached enough images within the loop
                    # The first condition counter > maximum_augmentation is optional
                    # because if the images that do not meet the max, it will be even
                    # and it will break in the other catch if statement, however, it's good
                    # to keep it for now

                    # print(f"\nBroke when counter was {counter}, and image_to_augment was {images_to_augment}\n")
                    # print(f"Reached the maximum amount of augmentation at count {counter}")
                    break
                elif switch == -1: #  Case 1: only rotations
                    augmented_img = Data_augmentation(image_path, theta_local, fact_local, False)
                    cv.imwrite(new_destination, augmented_img)
                    counter+=1
                elif switch == 1:  # Case 2: only horizontal flipped rotations
                    augmented_img = Data_augmentation(image_path, theta_local, fact_local, True)
                    cv.imwrite(new_destination, augmented_img)
                    counter+=1
            # Update tracking values
            switch = switch*-1
            if switch_ended % 2 == 0:
                theta_local+=theta
                fact_local+=(fact-1)  # --> these THETA and FACT values ensure no black padding on final image
            switch_ended+=1
            # Check again before continuing the while loop
            if counter > maximum_augmentation or counter > images_to_augment:
                total_augmented_images+=(counter-1)
                # print(f"Reached the maximum amount of augmentation at count {counter}")
                break  # This will break the while loop as well

    print(f"\n SiteIDBalancer() is completed for label {label}!\n")
    print(f"\n Original dataset contained {total_files} number of images\n")
    print(f"\n Augmented dataset should contain {total_files + total_augmented_images} number of images\n")

def SiteIDBalancer(input_dir, output_dir, label, theta, fact, multiplier): 
    # Obtain image directory
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, input_dir)
    dest_dir = os.path.join(root_dir, output_dir)

    # Create the list with the path of each label
    all_label_dirs = [os.path.join(dataset_dir, str(i)) for i in range(1, label+1)]

    # Create the list with the path of destination directories
    dest_label_dirs = [os.path.join(dest_dir, f"L{i}") for i in range(1, label+1)]

    # Copy all the original files to the destination directory
    print(f"\nCopying original files to {dest_dir}...")
    max_workers = 6
    with ThreadPoolExecutor(max_workers=max_workers) as executor: 
        futures = []
        for i in range(LABEL): 
            path_to_copy = os.path.join(dest_label_dirs[i], str(i+1))
            future = executor.submit(Copy_dir, all_label_dirs[i], path_to_copy)
            futures.append(future)

        # Wait for all tasks to complete
        for future in as_completed(futures): 
            try: 
                result = future.result()
                print(f"Copy task completed successfully")
            except Exception as e: 
                print(f"\nError in directory copy: {e}")

    # Perform the data augmentation in parallel
    max_workers = 6
    print(f"\nBalancing and augmenting dataset. Give us 5 minutes, please be patient...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor: 
        futures = []
        for i in range(len(all_label_dirs)):
            path_to_augment = os.path.join(dest_label_dirs[i], str(i+1))
            future = executor.submit(DataBalancer, all_label_dirs[i], 
                            path_to_augment, 
                            theta=theta, 
                            fact=fact, 
                            label=i+1, 
                            seed=SEED, 
                            multiplier=multiplier)
            futures.append(future)
    for future in as_completed(futures):
        try:
            result = future.result()  # This will raise any exceptions caught in the thread
            print("\nParallel data augmentation task completed successfully.")
        except Exception as e:
            print(f"\nError in data augmentation: {e}")

if __name__ == "__main__":
    des="""
    ------------------------------------------
    - Site ID Balancer/Augmentation (for Diffusion Models) (Overview) -

    Balances and perform the necessary image augmentation
    (e.g. 5-degree rotations + horizontal flip) to river
    stream images from CT DEEP, to prepare the dataset for
    diffusion model training (one model per label). Since
    the dataset is unbalanced, training it directly (without augmentation), 
    will cause the Diffusion Model to generate more images from certain 
    site ids. We will compensate that by augmenting the dataset. 
    ------------------------------------------
    - The Augmentation Process -

    The default augmentations are rotations every 5 degrees (from 5-30) 
    with horizontal flip before moving to the next degree rotation. The 
    script will augment all site_id to reach the site_id with the maximum
    amount of images. If there's not enough images to reach the maximum 
    amount of images, it will simply stop after it runs out of augmentation. 
    This ensures the site_ids with less images are represented in the distribution.
    ------------------------------------------
    - How to Use -

    > in_dir: (required) directory containing the labeled folders (e.g. 1,2...6)
    > out_dir (optional): the output directory
    > labels (default = 3): the labels you want to balance/augment in the in_dir starting from 0. 
      If 3, labels are 1-3. If 6, labels are 1-6
    > theta (default=5): the angle of rotation. If you change theta, you must
      change fact as well. If you're using 5 degrees interval, then it takes 1.3x zoom factor on each 
      image to crop out the padding resulted from the rotation.
    > fact (default=1.3): the zoom factor, after rotation.
    > multiplier (default=6): how many times to rotate. So if it's 6, it will rotate 
      from 5-30 degrees with horizontal flip in between, with a final 13x multiplier
      per image
    ------------------------------------------
    """
    # Initialize the Parser
    parser = argparse.ArgumentParser(description=des.lstrip(" "),formatter_class=argparse.RawTextHelpFormatter)

    # Add the arguments
    parser.add_argument('--in_dir',type=str,help='input directory of images with labeled subfolders\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--labels',type=str, help='the labels you want to balance/augment in the in_dir starting from 0. If 3, labels are 1-3\t[3]')
    parser.add_argument('--theta',type=int,help='the angle of rotation\t[5]')
    parser.add_argument('--fact', type=int, help='the zoom factor, after rotation\t[1.3]')
    parser.add_argument('--multiplier',type=int,help='how many times to rotate\t[6]')
    args = parser.parse_args()

    if args.in_dir is not None:
        DATASET_NAME = args.in_dir
    else: raise IOError
    if args.out_dir is not None:
        DEST_DIR = os.path.join(args.out_dir, DEST_DIR)
    else: DEST_DIR
    if args.labels is not None:
        LABEL = args.labels
    else: LABEL = 3
    if args.theta is not None:
        THETA = args.theta
    else: THESE = 5
    if args.fact is not None:
        FACT = args.fact
    else: FACT = 1.3
    if args.multiplier is not None:
        MULTIPLIER = args.multiplier
    else: MULTIPLIER = 6

    params = {'in_dir':DATASET_NAME,'out_dir':DEST_DIR,
              'labels': LABEL, 'theta':THETA,'fact':FACT,
              'multiplier':MULTIPLIER}
    print('\nUsing params:%s'%params)

    # Call the function
    SiteIDBalancer(DATASET_NAME, DEST_DIR, LABEL, THETA, FACT, MULTIPLIER)
    print(f"\nFinish balancing/augmentating the dataset\nCheck your directory for '{DEST_DIR}'")















