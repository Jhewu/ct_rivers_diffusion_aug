"""
THIS PYTHON FILE ONLY CONTAINS THE PARAMETERS
FOR THE DIFFUSION MODELS. THIS IS DONE TO AVOID
CIRCULAR IMPORT. 
"""
class Config: 
    def __init__(self): 
        """
        REMINDER: If add anything to the config file, consider 
        adding the arguments to argparse if they are useful
        """
        #-GENERAL PARAMETERS--------------------------------------
        # The location of the dataset
        # The in_dir will only work if it contains one more folder
        # within it, with the images, e.g. in_dir/1/images.JPG, 
        self.in_dir = "diffusion_data/cat1"            
        self.out_dir = "results"   
        self.run_description = "Running it on all three dataset"

        #-PREPROCESSING-------------------------------------------
        # Seed for the dataset split
        self.seed = 42
        self.validation_split = 0
        
        #-TRAINING------------------------------------------------
        # Runtime = ["training", "inference", "inpainting"]
        self.runtime = "training"
        self.load_and_train = False

        # The amount of stochastic noise used during reverse diffusion
        # If 0, it's DDIM (deterministic), if 1, it's DDPM (stochastic)
        self.eta = 1
        self.image_size = (200, 600)

        # Optimization (Training)
        self.num_epochs = 300
        self.batch_size = 8
        self.ema = 0.999                # --> DO NOT CHANGE
        self.learning_rate = 1e-5
        self.weigth_decay = self.learning_rate/10
        self.use_mix_precision = False
        self.gpu_index = 0
        self.min_signal_rate = 0.01     # --> DO NOT CHANGE
        self.max_signal_rate = 0.95     # --> DO NOT CHANGE

        # U-Net architecture
        self.embedding_dims = 256
        self.widths = [64, 96, 128, 256]
        self.block_depth = 3
        self.attention_in_bottleneck = False
        self.attention_in_up_down_sample = False

        # Callback 
        self.checkpoint_monitor = "n_loss"
        self.early_stop_monitor = "n_loss"
        self.early_stop_min_delta = self.learning_rate/10
        self.early_stop_start_epoch = self.ceil(self.num_epochs/2)
        self.early_stop_patience = self.ceil(self.early_stop_start_epoch/4)
        self.generate_on_epoch = float('inf')

        # Inference parameters
        self.model_dir = "results/"   # ---> This parameter is also used for training, when load_and_train is True 
        self.images_to_generate = 5
        self.generate_diffusion_steps = 30

        # Inpainting parameters
        self.inpainting_dir = "inpainting_data"

        # Subprocess checker
        self.subprocess = False     # --> DO NOT CHANGE

    def ceil(self, value): 
        return int(value) + (value % 1 > 0)
