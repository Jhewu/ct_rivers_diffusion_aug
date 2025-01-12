"""
THIS PYTHON FILE ONLY CONTAINS THE PARAMETERS
FOR THE DIFFUSION MODELS. THIS IS DONE TO AVOID
CIRCULAR IMPORT. 
"""

class Config: 
    def __init__(self): 
        # General parameters
        self.dataset_name = "flow_large"                            
        self.folder_path = "all_exp/exp5"
        self.label = "L2"
        self.run_description = ""
        self.models_to_train = [1, 2, 3]

        # Preprocessing
        self.seed = 42
        self.validation_split = 0.15
        
        # Training parameters
        self.runtime = "training"
        self.load_and_train = False
        self.eta = 0.5
        self.image_size = (200, 600)

        # Optimization (Training)
        self.num_epochs = 1
        self.batch_size = 4
        self.ema = 0.999
        self.learning_rate = 2.5e-4
        self.weigth_decay = self.learning_rate/10
        self.use_mix_precision = True
        self.gpu_index = 0
        self.min_signal_rate = 0.01
        self.max_signal_rate = 0.95

        # U-Net architecture
        self.embedding_dims = 128
        self.widths = [8, 16, 32]
        self.block_depth = 2
        self.attention_in_bottleneck = False
        self.attention_in_up_down_sample = False

        # Callback 
        self.checkpoint_monitor = "n_loss"
        self.early_stop_monitor = "n_loss"
        self.early_stop_min_delta = self.learning_rate/10
        self.early_stop_patience = 25
        self.early_stop_start_epoch = 50
        self.generate_on_epoch = 100000

        # Inference parameters
        self.images_to_generate = 5
        self.generate_diffusion_steps = 30

        # Inpainting parameters
        self.inpainting_dir = "inpainting_data"
        self.mask_and_image_dir = "mask_and_image"
