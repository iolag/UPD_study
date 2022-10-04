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
    if config.test:
        contrasts = []
        mean_intensities = []
        sizes = []
        # total_mean_score = []
        intensities = []
        norm_intensities = []
        norm_intensities_r = []
        norm_intensities_g = []
        norm_intensities_b = []
        intensities_r = []
        intensities_g = []
        intensities_b = []
        size_norm = []
        size_les = []
        # forward pass the testloader to extract anomaly maps, scores, masks, labels
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
        if config.test:
            norm_intensity = [inp[torch.logical_and(msk == 0, inp > 0)].flatten(
            ) for inp, msk in zip(input[:, 0].unsqueeze(1), mask)]
            norm_intensities.append(torch.cat(norm_intensity))
            intensity = [inp[msk != 0].flatten() for inp, msk in zip(input, mask)]
            intensities.append(torch.cat(intensity))

            mean_intensity = torch.tensor([inp[msk != 0].mean() for inp, msk in zip(input, mask)])
            mean_intensities.append(mean_intensity)
            # Relative area of lesion
            lesion_area = torch.count_nonzero(mask, dim=(1, 2, 3))
            size_les.append(lesion_area)
            brain_area = torch.count_nonzero(torch.stack(
                [inp > inp.min() for inp in input]), dim=(1, 2, 3))
            size_norm.append(brain_area)
            sizes.append(lesion_area / brain_area)

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
    if config.test:
        sizes = torch.cat(sizes)
        print(torch.cat(size_les).sum())
        print(torch.cat(size_norm).sum())
        print(torch.count_nonzero(torch.cat(labels)), len(torch.cat(labels)))
        intensities = torch.cat(intensities)
        norm_intensities = torch.cat(norm_intensities)
        mean_intensities = torch.cat(mean_intensities)
        torch.save(
            mean_intensities,
            f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/mean_intensities_{config.sequence}_{config.brats_t1}_{config.method}_new.pt')

        torch.save(
            norm_intensities,
            f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/norm_intensities_{config.sequence}_{config.brats_t1}_{config.method}_new.pt')

        torch.save(
            intensities,
            f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/intensities_{config.sequence}_{config.brats_t1}_{config.method}_new.pt')
        torch.save(
            sizes,
            f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/sizes_{config.sequence}_{config.brats_t1}_{config.method}_new.pt')
        exit(0)
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
