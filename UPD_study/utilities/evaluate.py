from typing import Callable
import torch
from UPD_study.utilities.utils import metrics, log
from argparse import Namespace
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import perf_counter


def evaluate(config: Namespace, test_loader: DataLoader, val_step: Callable) -> None:
    """
    Common evaluation method. Handles inference on evaluation set, metric calculation,
    logging and the speed benchmark.

    Args:
        config (Namespace): configuration object.
        test_loader (DataLoader): evaluation set dataloader
        val_step (Callable): validation step function
    """

    labels = []
    anomaly_scores = []
    anomaly_maps = []
    inputs = []
    segmentations = []
    benchmark_step = 0
    total_elapsed_time = 0

    # forward pass the testloader to extract anomaly maps, scores, masks, labels
    for input, mask in tqdm(test_loader, desc="Test set", disable=config.speed_benchmark):

        if config.speed_benchmark:
            timer = perf_counter()

        # forward pass, output = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
        output = val_step(input.to(config.device), test_samples=True)
        inputs.append(input.cpu())
        anomaly_maps.append(output[0].cpu())
        anomaly_scores.append(output[1].cpu())
        segmentations.append(mask)

        label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
        labels.append(label)

        if config.speed_benchmark:
            benchmark_step += 1
            # ignore 3 warmup steps
            if benchmark_step > 3:
                run_time = perf_counter() - timer
                total_elapsed_time += run_time

            if benchmark_step == 13:
                print(f"last batch inference time: {run_time} - Mean batch inference time: ",
                      total_elapsed_time / (benchmark_step - 3),
                      "fps: ", 160 / total_elapsed_time)
                return
    print(torch.cat(anomaly_maps).max(), torch.cat(anomaly_maps).min())
    # calculate metrics like AP, AUROC, on pixel and/or image level
    metrics(config, anomaly_maps, segmentations, anomaly_scores, labels)

    # do a single forward pass to extract images to log
    # the batch size is num_images_log for test_loaders, so only a single forward pass necessary
    input, mask = next(iter(test_loader))

    # forward pass, output = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
    output = val_step(input.to(config.device), test_samples=True)

    anomaly_maps = output[0]

    log({'anom_val/input images': input,
        'anom_val/targets': mask,
         'anom_val/anomaly maps': anomaly_maps}, config)

    # if recon based method, len(x)==3 because val_step returns reconstructions.
    # if thats the case log the reconstructions
    if len(output) == 3:
        log({'anom_val/reconstructions': output[2]}, config)
    # else:
    #     i = 0
    #     for input, mask in iter(test_loader):
    #         #input, mask = next(iter(test_loader))
    #         i += 1
    #         config.step += 1
    #         # forward pass, output = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
    #         output = val_step(input.to(config.device), test_samples=True)

    #         anomaly_maps = output[0]

    #         log({'anom_val/input images': input,
    #             'anom_val/targets': mask,
    #              'anom_val/anomaly maps': anomaly_maps}, config)

    #         # if recon based method, len(x)==3 because val_step returns reconstructions.
    #         # if thats the case log the reconstructions
    #         if len(output) == 3:
    #             log({'anom_val/reconstructions': output[2]}, config)
    #         if i == 4:
    #             exit(0)
    # labels = []
    # anomaly_scores = []
    # anomaly_maps = []
    # inputs = []
    # segmentations = []
    # benchmark_step = 0
    # total_elapsed_time = 0

    # sizes = []

    # # forward pass the testloader to extract anomaly maps, scores, masks, labels
    # # forward pass the testloader to extract anomaly maps, scores, masks, labels
    # for input in tqdm(test_loader, desc="Test set", disable=config.speed_benchmark):

    #     if config.speed_benchmark:
    #         timer = perf_counter()

    #     # forward pass, output = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
    #     output = val_step(input.to(config.device), test_samples=True)
    #     inputs.append(input.cpu())
    #     anomaly_maps.append(output[0].cpu())
    #     anomaly_scores.append(output[1].cpu().numpy())
    #     # segmentations.append(mask)

    #    # label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
    #   #  labels.append(label)

    #     size = torch.sum(torch.where(input > 0., 1., 0.), dim=(1, 2, 3))
    #     sizes.append(size.cpu().numpy())
    # import numpy as np
    # from matplotlib import pyplot as plt
    # fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    # axs.hist2d(np.concatenate(sizes), np.concatenate(anomaly_scores), bins=100)
    # axs.set_xlabel('Size', fontsize=18)
    # axs.set_ylabel('Anomaly Score', fontsize=18)
    # m, b = np.polyfit(np.concatenate(sizes), np.concatenate(anomaly_scores), deg=1)
    # axs.axline(xy1=(0, b), slope=m, c='red')

    # fig.savefig('histograms.png', format='png')
