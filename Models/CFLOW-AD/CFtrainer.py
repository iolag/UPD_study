import sys
sys.path.append('/data_ssd/users/lagi/thesis/UAD_study/')
import torch
from argparse import ArgumentParser
import numpy as np
from time import time
import torch.nn.functional as F
from model import load_decoder, load_encoder, positionalencoding2d, activation
from Utilities.common_config import common_config
from Utilities.evaluate import evaluate
from Utilities.utils import (seed_everything, load_data, load_pretrained,
                             misc_settings, log)


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = ArgumentParser(description='CFLOW-AD')

    parser = common_config(parser)

    # Model Hyperparameters
    parser.add_argument('--arch', default='wide_resnet50_2', type=str, metavar='A',
                        help='feature extractor: wide_resnet50_2/resnet18')
    parser.add_argument('-pl', '--num-pool-layers', default=3, type=int, metavar='L',
                        help='number of layers used in NF model (default: 3)')
    parser.add_argument('-cb', '--coupling-blocks', default=8, type=int, metavar='L',
                        help='number of layers used in NF model (default: 8)')

    # Training Hyperparameters
    parser.add_argument('--max-steps', '-ms', type=int, default=3000, help='Number of training steps')
    parser.add_argument('-bs', '--batch-size', default=32, type=int, metavar='B',
                        help='train batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')

    return parser.parse_args()


config = get_config()

# network hyperparameters
config.clamp_alpha = 1.9  # see paper equation 2 for explanation
config.condition_vec = 128

config.method = 'CFLOW-AD'

# general setup
misc_settings(config)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""
# specific seed for same dataloader creation accross different seeds
seed_everything(42)

train_loader, val_loader, big_testloader, small_testloader = load_data(config)

""""""""""""""""""""""""""""" Other method settings """""""""""""""""""""""""""""

log_theta = torch.nn.LogSigmoid()
GCONST = -0.9189385332046727  # ln(sqrt(2*pi))


""""""""""""""""""""""""""" Init model/optimizer """""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

# Backbone pre-trained encoder
encoder, pool_layers, pool_dims = load_encoder(config)
encoder = encoder.to(config.device)


# load pretrained with CCD
if config.load_pretrained:
    encoder = load_pretrained(encoder, config)

encoder.eval()

# Normalizing Flows decoder
decoders = [load_decoder(config, pool_dim) for pool_dim in pool_dims]
decoders = [decoder.to(config.device) for decoder in decoders]

params = list(decoders[0].parameters())
for i in range(1, config.num_pool_layers):
    params += list(decoders[i].parameters())

# optimizer
optimizer = torch.optim.Adam(params, lr=config.lr)

if config.eval:
    [decoder.load_state_dict(torch.load(f'saved_models/{config.modality}/{config.name}_decoder_{i}.pth'))
     for i, decoder in enumerate(decoders)]


""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def train():

    print('Starting training CFLOW-AD...')

    train_losses = []
    t_start = time()

    while True:
        for i, batch in enumerate(train_loader):

            config.step += 1
            loss = train_step(batch)
            train_losses.append(loss.item())

            if config.step % config.log_frequency == 0:
                # Print training loss
                log_msg = f" Train Loss: {np.mean(train_losses):.4f}"
                log_msg = f"Iteration {config.step} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"

                print(log_msg)

                # Log to wandb
                log({'train/loss': np.mean(train_losses)}, config)

                # Reset loss
                train_losses = []

            if config.step % config.anom_val_frequency == 0 or config.step == 100 or config.step == 500:
                evaluate(config, small_testloader, eval_step, val_loader)

            if config.step >= config.max_steps:
                print(f'Reached {config.max_steps} iterations. Finished training {config.name}.')
                for i, decoder in enumerate(decoders):
                    torch.save(decoder.state_dict(),
                               f'saved_models/{config.modality}/{config.name}_decoder_{i}.pth')
                return

            if config.step % config.save_frequency == 0:
                for i, decoder in enumerate(decoders):
                    torch.save(decoder.state_dict(),
                               f'saved_models/{config.modality}/{config.name}_{config.step}_decoder_{i}.pth')


def train_step(batch):
    batch = batch.to(config.device)

    # if grayscale repeat channel dim
    if batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)

    # forward pass to hook feature maps in activation dict
    with torch.no_grad():
        _ = encoder(batch)

    [decoder.train() for decoder in decoders]
    optimizer.zero_grad()

    train_loss = 0.0
    train_count = 0
    train_dist = []
    # iterate over specified {num_pool_layers} number of activations and train a decoder for each
    for i, layer in enumerate(pool_layers):  # eg. layer = 'layer 1' (lower number means deeper block)

        decoder = decoders[i]

        # "activation" dict aggregates activations after forward pass
        feature_map = activation[layer].detach()

        B, C, H, W = feature_map.size()
        BHW = B * H * W
        HW = H * W
        cond_vec = config.condition_vec

        # create b*h*w number of feature vectors with C features.
        e_r = feature_map.reshape(B, C, HW).transpose(1, 2).reshape(BHW, C)  # [BHW,C]

        # Spatial information is lost above, hence spatial prior is incorporated
        # with b*h*w conditional vectors in c_r, one for each feature vector of e_r
        p = positionalencoding2d(cond_vec, H, W).to(config.device).unsqueeze(0).repeat(B, 1, 1, 1)
        c_r = p.reshape(B, cond_vec, HW).transpose(1, 2).reshape(BHW, cond_vec)  # [BHW, cond_vec}

        z, log_jac_det = decoder(e_r, [c_r, ])
        decoder_log_prob = C * GCONST - 0.5 * torch.sum(z**2, 1) + log_jac_det
        log_prob = decoder_log_prob / C  # likelihood per dim
        loss = -log_theta(log_prob)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        train_loss += loss.sum().cpu().data.numpy()
        train_count += len(loss)
        train_dist.append(log_prob.detach())

    config.current_max = torch.tensor([0, 0, 0]).to(config.device)

    for i, p in enumerate(pool_layers):
        test_prob = train_dist[i]
        layer_max = test_prob.max()
        if layer_max > config.current_max[i]:
            config.current_max[i] = layer_max

    mean_train_loss = train_loss / train_count

    return mean_train_loss


@torch.no_grad()
def eval_step(input, return_loss: bool = False):
    """Forward-pass images into the network to extract encoder features and compute probability.
        Args:
          input: Batch of images.
        Returns:
          anomaly_map, anomaly_score: Predicted anomaly maps and scores.
    """
    # Compute anomaly map
    input = input.to(config.device)

    # if grayscale repeat channel dim
    if input.shape[1] == 1:
        input = input.repeat(1, 3, 1, 1)

    # Forward pass to extract features to "activation" dict
    _ = encoder(input)

    [decoder.eval() for decoder in decoders]

    height = []
    width = []
    test_dist = []  # [list() for layer in pool_layers]

    for i, layer in enumerate(pool_layers):
        decoder = decoders[i]
        feature_map = activation[layer]
        B, C, H, W = feature_map.size()
        BHW = B * H * W
        HW = H * W

        # get h, w of pool layers to be used when reshaping vectors bellow
        height.append(H)
        width.append(W)

        p = positionalencoding2d(config.condition_vec, H, W).to(config.device).unsqueeze(0).repeat(B, 1, 1, 1)
        c_r = p.reshape(B, config.condition_vec, HW).transpose(
            1, 2).reshape(BHW, config.condition_vec)  # BHWxP
        e_r = feature_map.reshape(B, C, HW).transpose(1, 2).reshape(BHW, C)  # BHWxC

        z, log_jac_det = decoder(e_r, [c_r, ])
        decoder_log_prob = C * GCONST - 0.5 * torch.sum(z**2, 1) + log_jac_det
        log_prob = decoder_log_prob / C  # likelihood per dim

        test_dist.append(log_prob.detach())

    # will use this during test time to infer a max to normalize (see __main__)
    config.test_max = torch.tensor([0, 0, 0]).to(config.device)
    for i, p in enumerate(pool_layers):
        test_prob = test_dist[i]
        layer_max = test_prob.max()
        if layer_max > config.current_max[i]:
            config.current_max[i] = layer_max

    test_map = [list() for p in pool_layers]

    for i, p in enumerate(pool_layers):

        test_prob = test_dist[i]  # EHWx1
        test_prob = test_prob.reshape(-1, height[i], width[i])
        # normalize likelihoods to (-Inf:0] by subtracting a constant
        test_prob = test_prob - torch.max(config.current_max)  # -est_prob.max()  #
        test_prob = torch.exp(test_prob)  # convert to probs in range [0:1]
        #print(test_prob.min(), test_prob.max())
        # upsample
        test_map[i] = F.interpolate(test_prob.unsqueeze(1),
                                    size=config.image_size, mode='bilinear',
                                    align_corners=True).cuda()

    # score aggregation
    anomaly_map = torch.zeros_like(test_map[0])
    for i, p in enumerate(pool_layers):
        anomaly_map += test_map[i]
    anomaly_map /= len(pool_layers)
    # invert to get anomaly maps
    # changed this to not be depend on batch and also have same scaling
    #  (instead of per batch sample anom_map.max() - anom_map)
   # print(test_prob.min(), test_prob.max())
    anomaly_map = 1 - anomaly_map
    # p rint(anomaly_map.min(), anomaly_map.max())

    # apply brainmask for MRI
    if config.modality in ['MRI', 'MRInoram', 'CT']:
        # normalize brain pixels only
        input = input[:, 0].unsqueeze(1)
        mask = torch.stack([inp > inp.min() for inp in input])
        # scale anomaly maps by substracting the minimum non-background value for better visualisation
        anomaly_map *= mask
        # mins = [(map[msk].min()) for map, msk in zip(anomaly_map, mask)]
        # anomaly_map = torch.cat([(map - min) for map, min in zip(anomaly_map, mins)]).unsqueeze(1)
        # anomaly_map *= mask

        anomaly_score = torch.tensor([map[inp > inp.min()].mean() for map, inp in zip(anomaly_map, input)])

    elif config.modality in ['RF']:
        mask = torch.stack([inp.mean(0, keepdim=True) > inp.mean(0, keepdim=True).min() for inp in input])
        anomaly_map *= mask
        anomaly_score = torch.tensor([map[inp.mean(0, keepdim=True) > inp.mean(0, keepdim=True).min()].mean()
                                      for map, inp in zip(anomaly_map, input)])
    else:
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])

    return anomaly_map, anomaly_score


if __name__ == '__main__':
    if not config.eval:
        train()
    else:
        print('Evaluating model...')
        # Do a train step to get config.current_max
        input = next(iter(train_loader))
        input = input.to(config.device)
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        config.current_max = torch.tensor([0, 0, 0]).to(config.device)

        _, _ = eval_step(input)
        config.current_max = config.test_max

        evaluate(config, big_testloader, eval_step, val_loader)


# perm = torch.randperm(BHW).to(config.device)  # BHW
# N = 64  # hyperparameter that increases batch size for the decoder model by N
# FIB = BHW // N  # number of fiber batches
# #https://github.com/gudovskiy/cflow-ad/issues/18
# #https://github.com/gudovskiy/cflow-ad/issues/18
# assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
# for f in range(FIB):  # per-fiber processing
#     idx = torch.arange(f * N, (f + 1) * N)
#     c_p = c_r[perm[idx]]  # NxP
#     e_p = e_r[perm[idx]]  # NxC

# if i_epoch % config.sub_epochs:
#     adjust_learning_rate(config, optimizer, i_epoch)
#     i_sub_epoch = 0

# config.lr = warmup_learning_rate(config, i_epoch, i + i_sub_epoch * len(train_loader),
#                                  len(train_loader) * config.sub_epochs, optimizer)

# # learning rate hparams
# config.lr_decay_epochs = [i * config.meta_epochs // 100 for i in [50, 75, 90]]
# print('LR schedule: {}'.format(config.lr_decay_epochs))
# config.lr_decay_rate = 0.1
# config.lr_warm_epochs = 2
# config.lr_warm = True
# config.lr_cosine = True
# if config.lr_warm:
#     config.lr_warmup_from = config.lr / 10.0
#     if config.lr_cosine:
#         eta_min = config.lr * (config.lr_decay_rate ** 3)
#         config.lr_warmup_to = eta_min + (config.lr - eta_min) * (
#             1 + math.cos(math.pi * config.lr_warm_epochs / config.meta_epochs)) / 2
#     else:
#         config.lr_warmup_to = config.lr
