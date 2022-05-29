import sys
sys.path.append('/home/ioannis/lagi/thesis')
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, resnet18
from Utilities.utils import seed_everything, metrics, load_data, load_pretrained, misc_settings
from Utilities.common_config import common_config
import wandb
os.environ["WANDB_SILENT"] = "true"

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = argparse.ArgumentParser()
    parser = common_config(parser)
    parser.add_argument('--arch', type=str, default='wide_resnet50_2',
                        choices=['resnet18', 'wide_resnet50_2'])
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

    return parser.parse_args()


config = get_config()

msg = "num_images_log should be lower or equal to batch size"
assert (config.batch_size >= config.num_images_log), msg

# Select training device
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get naming string
config.method = 'PADIM'
naming_str, _ = misc_settings(config)

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for deterministic dataloader creation
seed_everything(42)

if not config.eval:
    train_loader, val_loader, big_testloader, small_testloader = load_data(config)
else:
    big_testloader, small_testloader = load_data(config)

""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
print("Initializing Model...")

if config.arch == 'resnet18':
    model = resnet18(pretrained=True, progress=True)
    t_d = 448  # total channel dims of extracted embed. volume
    d = 100  # number of channels to subsample to


elif config.arch == 'wide_resnet50_2':
    model = wide_resnet50_2(pretrained=True, progress=True)
    t_d = 1792  # 256 512 1024
    d = 550

if config.load_pretrained:
    model = load_pretrained(model, config)

# create index vector to subsample embedding_vector
idx = torch.tensor(sample(range(0, t_d), d))

model.to(config.device)
model.eval()


print(f"{config.arch} initialized.")

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def train():
    # forward hook first 3 layer final activations
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    a = model.layer1.register_forward_hook(hook)
    b = model.layer2.register_forward_hook(hook)
    c = model.layer3.register_forward_hook(hook)

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # extract train set features
    for batch in tqdm(train_loader, '| feature extraction | train | %s |' % config.modality):

        # if grayscale repeat channel dim
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)

        # model prediction
        with torch.no_grad():
            _ = model(batch.to(config.device))

        # get intermediate layer outputs
        for k, v in zip(train_outputs.keys(), outputs):
            train_outputs[k].append(v.cpu())

        # reset hook outputs
        outputs = []

    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)

    print('Constructing embedding volume...')

    embedding_vectors = train_outputs['layer1']

    for layer_name in ['layer2', 'layer3']:
        next_layer_upscaled = F.interpolate(train_outputs[layer_name],
                                            size=embedding_vectors.size(-1),
                                            mode='bilinear',
                                            align_corners=False)

        embedding_vectors = torch.cat([embedding_vectors, next_layer_upscaled], dim=1)

    # randomly select d dimensions of embedding vector
    embedding_vectors = torch.index_select(embedding_vectors.cpu(), 1, idx)

    print('Calculating statistics...')
    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0).numpy()
    cov = torch.zeros(C, C, H * W).numpy()
    ident = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * ident

    # save learned distribution
    train_outputs = [mean, cov]
    with open(f'saved_models/{config.modality}/{config.arch}_{naming_str}.pkl', 'wb') as f:
        pickle.dump(train_outputs, f)

    a.remove()
    b.remove()
    c.remove()


def test(logger):

    # forward hook first 3 layer final activations
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    a = model.layer1.register_forward_hook(hook)
    b = model.layer2.register_forward_hook(hook)
    c = model.layer3.register_forward_hook(hook)

    # load saved statistics
    with open(f'saved_models/{config.modality}/{config.arch}_{naming_str}.pkl', 'rb') as f:
        train_outputs = pickle.load(f)

    labels = []
    anomaly_maps = []
    segmentations = []
    inputs = []
    anomaly_maps = []
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    # extract test set features
    for input, mask in tqdm(big_testloader, '| feature extraction | test | %s' % config.modality):

        inputs.append(input)
        label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
        labels.append(label)
        segmentations.append(mask)

        # if grayscale repeat channel dim
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)

        # model prediction
        with torch.no_grad():
            _ = model(input.to(config.device))

        # get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.detach())

        # reset hook outputs
        outputs = []

        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            next_layer_upscaled = F.interpolate(test_outputs[layer_name],
                                                size=embedding_vectors.size(-1),
                                                mode='bilinear',
                                                align_corners=False)

            embedding_vectors = torch.cat([embedding_vectors, next_layer_upscaled], dim=1)

        # reset test_outputs dict
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # randomly select d dimensions of embedding vector
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx.to(config.device))

        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).cpu().numpy()
        dist_list = []
        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample anoamaly maps
        dist_list = torch.tensor(dist_list)
        anomaly_map = F.interpolate(dist_list.unsqueeze(1), size=input.size(2), mode='bilinear',
                                    align_corners=False).squeeze().numpy()
        # apply gaussian smoothing on the score map
        for i in range(anomaly_map.shape[0]):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)  # [samples, h , w]
        # https://pytorch.org/vision/stable/generated/torchvision.transforms.GaussianBlur.html
        anomaly_maps.append(anomaly_map)

    anomaly_maps = np.concatenate(anomaly_maps)

    evaluation(inputs, segmentations, labels, anomaly_maps, logger)

    a.remove()
    b.remove()
    c.remove()

    return


def evaluation(inputs, segmentations, labels, anomaly_maps, logger):

    # hacky stuff to get it compatible with my function metrics()
    anomaly_maps = torch.from_numpy(anomaly_maps)
    s, h, w = anomaly_maps.shape
    anomaly_maps = anomaly_maps.reshape(s, 1, 1, h, w)
    anomaly_maps = [map for map in anomaly_maps]
    inputs = torch.cat(inputs)  # tensor of shape num_samples,1,h,w
    inputs = inputs.reshape(s, 1, 1, h, w)
    inputs = [inp for inp in inputs]

    # apply brainmask for MRI
    if config.modality == 'MRI':
        masks = [inp > inp.min() for inp in inputs]
        anomaly_maps = [map * mask for map, mask in zip(anomaly_maps, masks)]

    anomaly_scores = [torch.Tensor([map[inp > inp.min()].mean() for map, inp in zip(anomaly_maps, inputs)])]
    threshold = metrics(anomaly_maps, segmentations, anomaly_scores,
                        labels, logger, 0, limited_metrics=False)

    # Log images to wandb
    input_images = list(inputs[:config.num_images_log])
    targets = list(segmentations[0].float()[:config.num_images_log])
    anomaly_images = anomaly_maps[:config.num_images_log]

    # create thresholded images on the threshold of best posible dice score on the dataset
    anomaly_thresh_map = torch.cat(anomaly_maps)[:config.num_images_log]
    anomaly_thresh_map = torch.where(anomaly_thresh_map < threshold,
                                     torch.FloatTensor([0.]),
                                     torch.FloatTensor([1.]))

    anomaly_thresh = list(anomaly_thresh_map)

    logger.log({
        'anom_val/input images': [wandb.Image(img) for img in input_images],
        'anom_val/targets': [wandb.Image(img) for img in targets],
        'anom_val/anomaly maps': [wandb.Image(img) for img in anomaly_images],
        'anom_val/thresholded maps': [wandb.Image(img) for img in anomaly_thresh],
    }, step=0)


if __name__ == '__main__':

    if not config.eval:
        train()
    else:
        logger = wandb.init(project='PADIM', name=naming_str, config=config, reinit=True)
        test(logger)
