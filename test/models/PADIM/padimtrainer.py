import sys
import os
sys.path.append(os.path.expanduser('~/thesis/UAD_study/'))
from random import sample
import argparse
import numpy as np
import pickle
from tqdm import tqdm
from collections import OrderedDict
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
# from torchvision.models import wide_resnet50_2, resnet18
from resnet import wide_resnet50_2, resnet18
from Utilities.utils import seed_everything, metrics, load_data, load_pretrained, misc_settings, log
from Utilities.common_config import common_config
import wandb

from Utilities.normFPR import dice_f1_normal_nfpr

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = argparse.ArgumentParser()
    parser = common_config(parser)
    parser.add_argument('--arch', type=str, default='wide_resnet50_2',
                        choices=['resnet18', 'wide_resnet50_2'])
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

    return parser.parse_args()


config = get_config()

# get naming string
config.method = 'PADIM'
if not config.get_images:
    config.disable_wandb = True
misc_settings(config)

print(config.name)
if config.eval and not config.get_images:
    logger = wandb.init(project='PADIM', name=config.name, config=config, reinit=True)
    config.logger = logger

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""
if config.modality == 'MRI' and not config.eval:
    config.normal_split = 0.2


# specific seed for deterministic dataloader creation
seed_everything(42)

train_loader, val_loader, big_testloader, small_testloader = load_data(config)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

print("Initializing Model...")

if config.arch == 'resnet18':
    model = resnet18(pretrained=True)
    t_d = 448  # total channel dims of extracted embed. volume
    d = 100  # number of channels to subsample to


elif config.arch == 'wide_resnet50_2':
    model = wide_resnet50_2(pretrained=True)
    t_d = 1792  # 256 512 1024
    d = 550

if config.load_pretrained:
    model = load_pretrained(model, config)

# create index vector to subsample embedding_vector
idx = torch.tensor(sample(range(0, t_d), d))

model.to(config.device)
model.eval()


print(f"{config.arch} initialized.")

if config.speed_benchmark:
    from torchinfo import summary
    a = summary(model, (16, 3, 128, 128), verbose=0)
    params = a.total_params
    macs = a.total_mult_adds
    print('Number of Million parameters: ', params / 1e06)
    print('Number of GMACs: ', macs / 1e09)
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
    with open(f'saved_models/{config.modality}/{config.arch}_{config.name}.pkl', 'wb') as f:
        pickle.dump(train_outputs, f)

    a.remove()
    b.remove()
    c.remove()


""""""""""""""""""""""""""""""""""" Testing """""""""""""""""""""""""""""""""""

from time import perf_counter


def test(dataloader, normal_test: bool = False):

    # forward hook first 3 layer final activations
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    a = model.layer1.register_forward_hook(hook)
    b = model.layer2.register_forward_hook(hook)
    c = model.layer3.register_forward_hook(hook)

    # load saved statistics
    with open(f'saved_models/{config.modality}/{config.arch}_{config.name}.pkl', 'rb') as f:
        train_outputs = pickle.load(f)

    labels = []
    anomaly_maps = []
    segmentations = []
    inputs = []
    anomaly_maps = []
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    # extract test set features
    total_elapsed_time = 0
    benchmark_step = 0
    for batch in tqdm(dataloader, '| feature extraction | test | %s' % config.modality):
        timer = perf_counter()
        if normal_test:  # for normal fpr calc we pass normal set
            input = batch
            inputs.append(input)
        else:
            input = batch[0]
            mask = batch[1]
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
            # scipy.spatial.distance.mahalanobis takes in the inverse of the covariance matrix
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample anoamaly maps
        dist_list = torch.tensor(dist_list)
        anomaly_map = F.interpolate(dist_list.unsqueeze(1), size=input.size(2), mode='bilinear',
                                    align_corners=False).squeeze().numpy()
        # apply gaussian smoothing on the score map
        if config.gaussian_blur:
            for i in range(anomaly_map.shape[0]):
                anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)  # [samples, h , w]

        # noticed that when a single sample batch occures, anomaly_map loses its batch dim somewhere above
        if len(anomaly_map.shape) == 2:
            anomaly_map = np.expand_dims(anomaly_map, 0)

        anomaly_maps.append(anomaly_map)

        if config.speed_benchmark:
            benchmark_step += 1
            # ignore 3 warmup steps
            if benchmark_step > 3:
                run_time = perf_counter() - timer
                total_elapsed_time += run_time

                if benchmark_step == 13:
                    print(f"current forward pass time: {run_time} - Mean inference time: ",
                          total_elapsed_time / (benchmark_step - 3),
                          "fps: ", 160 / total_elapsed_time)
                    exit(0)

    anomaly_maps = np.concatenate(anomaly_maps)

    a.remove()
    b.remove()
    c.remove()

    # hacky stuff to get it compatible with my function metrics()
    anomaly_maps = torch.from_numpy(anomaly_maps)
    s, h, w = anomaly_maps.shape
    anomaly_maps = anomaly_maps.reshape(s, 1, 1, h, w)
    anomaly_maps = [map for map in anomaly_maps]
    inputs2 = torch.cat(inputs)  # tensor of shape num_samples,c,h,w
    if config.modality in ['RF', 'COL']:  # RGB case
        inputs2 = inputs2.reshape(s, 1, 3, h, w)
    else:
        inputs2 = inputs2.reshape(s, 1, 1, h, w)
    inputs2 = [inp for inp in inputs2]
    #

    # apply brainmask for MRI, RF
    if config.modality in ['MRI', 'CT']:
        masks = [inp > inp.min() for inp in inputs2]

        if config.get_images:
            anomaly_maps = [map * mask for map, mask in zip(anomaly_maps, masks)]
            mins = [(map[map > 0]) for map in anomaly_maps]
            mins = [map.min() for map in mins]

            anomaly_maps = torch.cat([(map - min) for map, min in zip(anomaly_maps, mins)]).unsqueeze(1)

        anomaly_maps = [map * mask for map, mask in zip(anomaly_maps, masks)]
        anomaly_scores = [torch.Tensor([map[inp > inp.min()].max()
                                        for map, inp in zip(anomaly_maps, inputs2)])]

    elif config.modality in ['RF'] and config.dataset == 'DDR':
        anomaly_scores = [torch.Tensor([map.max() for map in anomaly_maps])]
    else:
        anomaly_scores = [torch.tensor([map.mean() for map in anomaly_maps])]

    return inputs, segmentations, labels, anomaly_maps, anomaly_scores


def evaluation(inputs, segmentations, labels, anomaly_maps, anomaly_scores):

    # or not config.get_images:
    if config.modality in ['CXR', 'OCT'] or (config.dataset != 'DDR' and config.modality == 'RF'):
        segmentations = None  # disables pixel level evaluation in metrics()
    if not config.get_images:
        threshold = metrics(config, anomaly_maps, segmentations, anomaly_scores, labels)

    # Log images to wandb
    #targets = [inp.squeeze(0) for inp in targets]
    anomaly_maps = torch.cat(anomaly_maps)[:config.num_images_log]

    # print(torch.stack(anomaly_maps)[:20].shape)
    if config.get_images:
        list_to_log = list(inputs[0][:config.num_images_log].float().cpu())
    else:
        list_to_log = list(inputs[0][:config.num_images_log].float().cpu())
    # print(len(list_to_log))
    # config.logger.log({
    #     'anom_val/input images': [wandb.Image(img) for img in list_to_log]
    # }, step=config.step)
    log({'anom_val/input images': inputs[0],
         'anom_val/anomaly maps': anomaly_maps}, config)
    # 'anom_val/targets': segmentations[0]

    # if not config.limited_metrics and segmentations is not None:
    #     # create thresholded images on the threshold of best posible dice score on the dataset

    #     anomaly_thresh_map = torch.cat(anomaly_maps)[:config.num_images_log]
    #     anomaly_thresh_map = torch.where(anomaly_thresh_map < threshold,
    #                                      torch.FloatTensor([0.]),
    #                                      torch.FloatTensor([1.]))

    #     anomaly_thresh = list(anomaly_thresh_map)

    #     log({'anom_val/thresholded maps': anomaly_thresh}, config)


if __name__ == '__main__':

    if not config.eval:
        train()
    else:
        evaluation(*test(big_testloader))
