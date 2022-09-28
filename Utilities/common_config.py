from Utilities.utils import str_to_bool


def common_config(parser):
    parser.add_argument('--test', type=str_to_bool, default=False,
                        help='percentage of samples to use for percentage experiment')
    parser.add_argument('--atlas_exp', type=str_to_bool, default=False,
                        help='percentage of samples to use for percentage experiment')
    parser.add_argument('--percentage', '-pc', type=int, default=100,
                        help='percentage of samples to use for percentage experiment')
    parser.add_argument('--sigma', '-s', type=float, default=4, help='sigma of gaussian filter')
    parser.add_argument('--shuffle', '-sh', type=str_to_bool, default=False,
                        help='shuffle test set (after splitting)')
    parser.add_argument('--early-validation', '-early', type=str_to_bool, default=False, help='')
    parser.add_argument('--deep', '-dp', type=str_to_bool, default=False, help='')
    parser.add_argument('--get_images', '-img', type=str_to_bool, default=False, help='')
    parser.add_argument('--speed_benchmark', '-sb', type=str_to_bool, default=False, help='')
    parser.add_argument('--gaussian-blur', '-gb', type=str_to_bool, default=False, help='')
    parser.add_argument('--small-is-big', '-sib', type=str_to_bool, default=False, help='')
    parser.add_argument('--patches', type=str_to_bool, default=False, help='')
    parser.add_argument('--num_patches', '-np', type=int, default=9,
                        help='', choices=[1, 4, 9, 16, 25])

    parser.add_argument('--norm_vol', type=str_to_bool, default=False,
                        help='Load encoder pretrained with CCD')
    # Normal FPR metric settings
    parser.add_argument('--normal_fpr', type=str_to_bool, default=False, help='Implement normal fpr metric')
    parser.add_argument('--dice_normal', type=str_to_bool, default=False, help='Implement normal fpr metric')
    parser.add_argument('--f1_normal', type=str_to_bool, default=False, help='Implement normal fpr metric')
    parser.add_argument('--nfpr', type=float, default=0.05, help='fpr for normal fpr metric')

    # General script settings
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--load_pretrained', '-lp', type=str_to_bool, default=False,
                        help='Load encoder pretrained with CCD')
    parser.add_argument('--disable_wandb', '-dw', type=str_to_bool,
                        default=False, help='disable wandb logging')
    parser.add_argument('--eval', '-ev', type=str_to_bool, default=False, help='Evaluation mode')
    parser.add_argument('--print_model', type=str_to_bool, default=False,
                        help='Print model information with torchsummary.summary')
    parser.add_argument('--limited_metrics', '-lm', type=str_to_bool, default=False,
                        help='onlt return image-level metrics and pixel AP')
    # Data settings
    parser.add_argument('--datasets_dir', type=str,
                        default='Datasets', help='datasets_dir')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--img_channels', type=int, default=1, help='Image channels')
    parser.add_argument('--center', '-cnt', type=str_to_bool, default=False,
                        help='Whether to center the samples to [-1,1] range.')
    parser.add_argument('--stadardize', '-stad', type=str_to_bool, default=False,
                        help='Whether to standardize the samples to N(0,1) dataset-wise.')
    parser.add_argument('--modality', '-mod', type=str, default='MRI', help='MRI sequence')
    parser.add_argument('--normal_split', '-ns', type=float, default=0.95, help='normal set split')
    parser.add_argument('--anomal_split', '-as', type=float, default=0.90, help='anomaly set split')
    parser.add_argument('--num_workers', type=int, default=30, help='Number of workers')

    # MRI specific settings
    parser.add_argument('--sequence', '-seq', type=str, default='t2',
                        help='MRI sequence', choices=['t1', 't2', 't1+t2'])
    parser.add_argument('--brats_t1', type=str_to_bool, default=True)
    parser.add_argument('--slice_range', type=int, nargs='+',
                        default=(0, 155), help='Lower and Upper slice index')
    parser.add_argument('--normalize', type=str_to_bool, default=False,
                        help='Normalize images to 98th percentile and scale to [0,1]')
    parser.add_argument('--equalize_histogram', type=str_to_bool,
                        default=False, help='Equalize histogram')
    parser.add_argument('--histogram_matching', '-match', type=str_to_bool,
                        default=False, help='Match histogram with a reference slice.')
    # CXR specific settings
    parser.add_argument('--sup_devices', type=str_to_bool, default=False,
                        help='Whether to include CXRs with support devices')
    parser.add_argument('--AP_only', type=str_to_bool, default=True, help='Whether to include only AP CXRs')
    parser.add_argument('--pathology', type=str, default='effusion',
                        help='Pathology of test set.', choices=['enlarged', 'effusion', 'opacity'])
    parser.add_argument('--sex', type=str, default='both',
                        help='Sex of patients', choices=['male', 'female', 'both'])

    # RF specific settings
    parser.add_argument('--dataset', type=str, default='DDR',
                        help='Which dataset to use.', choices=['KAGGLE', 'LAG', 'IDRID', 'DDR'])

    # Logging settings
    parser.add_argument('--name_add', '-nam', type=str, default='', help='option to add to the wandb name')
    parser.add_argument('--log_frequency', '-lf', type=int, default=200, help='logging frequency')
    parser.add_argument('--val_frequency', '-vf', type=int, default=1000, help='validation frequency')
    parser.add_argument('--anom_val_frequency', '-avf', type=int, default=1000,
                        help='Validation frequency on anomalous samples')
    parser.add_argument('--val_steps', type=int, default=50, help='validation steps')
    parser.add_argument('--num_images_log', '-nil', type=int, default=32, help='Number of images to log')

    # SSIM evaluation
    parser.add_argument('--ssim_eval', '-ssim', type=str_to_bool, default=True,
                        help='SSIM for reconstruction residual')

    # Save, Load, Train part settings
    parser.add_argument('--save_frequency', type=int, default=100000, help='model save/checkpoint frequency')
    # parser.add_argument('--load_saved', type=str_to_bool, default=False, help='load a saved model')
    parser.add_argument('--load_iter', type=str, default="", help='iteration/checkpoint of model to load')

    return parser
