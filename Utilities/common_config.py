import sys
sys.path.append('/home/ioannis/lagi/thesis/')
from Utilities.utils import str_to_bool


def common_config(parser):

    # General script settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--load_pretrained', type=str_to_bool, default=False,
                        help='Load encoder pretrained with CCD')
    parser.add_argument('--disable_wandb', type=str_to_bool, default=False, help='disable wandb logging')
    parser.add_argument('--eval', type=str_to_bool, default=False, help='Evaluation mode')
    parser.add_argument('--print_model', type=str_to_bool, default=False,
                        help='Print model information with torchsummary.summary')

    # Data settings
    parser.add_argument('--datasets_dir', type=str,
                        default='/datasets/Datasets/', help='datasets_dir')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--img_channels', type=int, default=1, help='Image channels')
    parser.add_argument('--stadardize', type=str_to_bool, default=False,
                        help='Whether to standardize the samples to N(0,1) dataset-wise.')
    parser.add_argument('--modality', '-mod', type=str, default='MRI', help='MRI sequence')
    parser.add_argument('--normal_split', '-ns', type=float, default=0.90, help='normal set split')
    parser.add_argument('--anomal_split', '-as', type=float, default=0.90, help='anomaly set split')
    parser.add_argument('--num_workers', type=int, default=7, help='Number of workers')

    # MRI specific settings
    parser.add_argument('--sequence', type=str, default='t1', help='MRI sequence', choices=['t1', 't2'])
    parser.add_argument('--slice_range', type=int, nargs='+',
                        default=(0, 155), help='Lower and Upper slice index')
    parser.add_argument('--normalize', type=str_to_bool, default=False,
                        help='Normalize images to 98th percentile and scale to [0,1]')
    parser.add_argument('--equalize_histogram', type=str_to_bool,
                        default=False, help='Equalize histogram')

    # CXR specific settings
    parser.add_argument('--sup_devices', type=str_to_bool, default=False,
                        help='Whether to include CXRs with support devices')
    parser.add_argument('--AP_only', type=str_to_bool, default=True, help='Whether to include only AP CXRs')
    parser.add_argument('--pathology', type=str, default='effusion',
                        help='Pathology of test set.', choices=['enlarged', 'effusion', 'opacity'])
    parser.add_argument('--sex', type=str, default='both',
                        help='Sex of patients', choices=['male', 'female', 'both'])

    # RF specific settings
    parser.add_argument('--dataset', type=str, default='KAGGLE',
                        help='Which dataset to use.', choices=['KAGGLE', 'LAG', 'IDRID'])

    # Logging settings
    parser.add_argument('--name_add', type=str, default='', help='option to add to the wandb name')
    parser.add_argument('--log_frequency', '-lf', type=int, default=100, help='logging frequency')
    parser.add_argument('--val_frequency', '-vf', type=int, default=400, help='validation frequency')
    parser.add_argument('--anom_val_frequency', '-avf', type=int, default=200,
                        help='Validation frequency on anomalous samples')
    parser.add_argument('--val_steps', type=int, default=50, help='validation steps')
    parser.add_argument('--num_images_log', '-nil', type=int, default=4, help='Number of images to log')

    # SSIM evaluation
    parser.add_argument('--ssim_eval', type=str_to_bool, default=True,
                        help='SSIM for reconstruction residual')

    # Save, Load, Train part settings
    parser.add_argument('--save_frequency', type=int, default=1000, help='model save frequency')
    parser.add_argument('--load_saved', type=str_to_bool, default=False, help='load a saved model')
    parser.add_argument('--saved_iter', type=int, default=12000, help='iteration when model was saved')

    return parser