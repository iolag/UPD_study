import sys
import os
sys.path.append(os.path.expanduser('~/thesis/UAD_study/'))
import torch
from Utilities.utils import metrics, log
from argparse import Namespace
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import perf_counter
from matplotlib import pyplot as plt


def evaluate(config: Namespace, test_loader: DataLoader,
             val_step: object, val_loader: DataLoader = None) -> None:
    """
    because we work we datasets of both with and without masks, code is structured in a
    way that no mask dataloaders also return masks. For healthy samples they return empty
    masks, and for anomalous samples they return non-empty masks. The code then works the
    same and the binary label is produced by label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
    """
    if not config.get_images:
        labels = []
        anomaly_scores = []
        anomaly_maps = []
        inputs = []
        segmentations = []
        if config.atlas_exp:
            contrasts = []
            mean_intensities = []
            sizes = []
            #total_mean_score = []
            mean_score_ratios = []
            intensities = []
            norm_intensities = []
            norm_intensities_r = []
            norm_intensities_g = []
            norm_intensities_b = []
            intensities_r = []
            intensities_g = []
            intensities_b = []
            # forward pass the testloader to extract anomaly maps, scores, masks, labels
        benchmark_step = 0
        total_elapsed_time = 0
        if config.atlas_exp:
            for input, mask in tqdm(test_loader, desc="Test set"):

                if config.speed_benchmark:
                    timer = perf_counter()

                # forward pass, x = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
                x = val_step(input.to(config.device), return_loss=False)

                anomaly_maps.append(x[0].cpu())
                anomaly_scores.append(x[1].cpu())
                segmentations.append(mask)

                # Contrast ΔΙ of input and atlas on the lession area
                # contrast = torch.abs(input - atlas)
                # contrast = torch.tensor([con[msk != 0].mean() for con, msk in zip(contrast, mask)])
                # contrasts.append(contrast)

                # intensity
                if config.modality == 'COL':

                    norm_intensity_r = torch.tensor(
                        [inp[0].unsqueeze(0)[msk == 0].mean() for inp, msk in zip(input, mask)])
                    norm_intensity_g = torch.tensor(
                        [inp[1].unsqueeze(0)[msk == 0].mean() for inp, msk in zip(input, mask)])
                    norm_intensity_b = torch.tensor(
                        [inp[2].unsqueeze(0)[msk == 0].mean() for inp, msk in zip(input, mask)])
                    norm_intensities_r.append(norm_intensity_r)
                    norm_intensities_g.append(norm_intensity_g)
                    norm_intensities_b.append(norm_intensity_b)
                    intensity_r = torch.tensor(
                        [inp[0].unsqueeze(0)[msk != 0].mean() for inp, msk in zip(input, mask)])
                    intensity_g = torch.tensor(
                        [inp[1].unsqueeze(0)[msk != 0].mean() for inp, msk in zip(input, mask)])
                    intensity_b = torch.tensor(
                        [inp[2].unsqueeze(0)[msk != 0].mean() for inp, msk in zip(input, mask)])
                    intensities_r.append(intensity_r)
                    intensities_g.append(intensity_g)
                    intensities_b.append(intensity_b)
                    lesion_area = torch.count_nonzero(mask, dim=(1, 2, 3))
                    sizes.append(lesion_area)
                if config.modality == 'RF':
                    input = input.mean(1, keepdims=True)
                    norm_intensity = [inp[msk == 0].flatten(
                    ) for inp, msk in zip(input[:, 0].unsqueeze(1), mask)]
                    norm_intensities.append(torch.cat(norm_intensity))
                    intensity = [inp[msk != 0].flatten() for inp, msk in zip(input, mask)]
                    intensities.append(torch.cat(intensity))

                    mean_intensity = torch.tensor([inp[msk != 0].mean() for inp, msk in zip(input, mask)])
                    mean_intensities.append(mean_intensity)
                    # Relative area of lesion
                    lesion_area = torch.count_nonzero(mask, dim=(1, 2, 3))
                    # brain_area = torch.count_nonzero(torch.stack(
                    #     [inp > inp.min() for inp in input]), dim=(1, 2, 3))
                    sizes.append(lesion_area)

                    # Mean score of lesion area / mean score of normal area
                    # les_score = torch.tensor([map[msk != 0].mean() for map, msk in zip(x[0].cpu(), mask)])
                    # norm_score = torch.tensor([map[torch.logical_and(msk == 0, inp > inp.min())].mean()
                    #                            for map, msk, inp in zip(x[0].cpu(), mask, input)])
                    # mean_score_ratios.append(les_score / norm_score)
                else:
                    norm_intensity = [inp[torch.logical_and(msk == 0, inp > 0)].flatten(
                    ) for inp, msk in zip(input[:, 0].unsqueeze(1), mask)]
                    norm_intensities.append(torch.cat(norm_intensity))
                    intensity = [inp[msk != 0].flatten() for inp, msk in zip(input, mask)]
                    intensities.append(torch.cat(intensity))

                    mean_intensity = torch.tensor([inp[msk != 0].mean() for inp, msk in zip(input, mask)])
                    mean_intensities.append(mean_intensity)
                    # Relative area of lesion
                    lesion_area = torch.count_nonzero(mask, dim=(1, 2, 3))
                    brain_area = torch.count_nonzero(torch.stack(
                        [inp > inp.min() for inp in input]), dim=(1, 2, 3))
                    sizes.append(lesion_area / brain_area)

                    # Mean score of lesion area / mean score of normal area
                    les_score = torch.tensor([map[msk != 0].mean() for map, msk in zip(x[0].cpu(), mask)])
                    norm_score = torch.tensor([map[torch.logical_and(msk == 0, inp > inp.min())].mean()
                                               for map, msk, inp in zip(x[0].cpu(), mask, input)])
                    mean_score_ratios.append(les_score / norm_score)

                label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
                labels.append(label)
        else:
            for input, mask in tqdm(test_loader, desc="Test set"):

                if config.speed_benchmark:
                    timer = perf_counter()

                # forward pass, x = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
                x = val_step(input.to(config.device), return_loss=False)
                inputs.append(input.cpu())
                anomaly_maps.append(x[0].cpu())
                anomaly_scores.append(x[1].cpu())
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
                        print(f"current forward pass time: {run_time} - Mean inference time: ",
                              total_elapsed_time / (benchmark_step - 3),
                              "fps: ", 160 / total_elapsed_time)
                        exit(0)
            if config.test:
                import numpy as np
                labels = torch.cat(labels)
                anomaly_scores = torch.cat(anomaly_scores)
                segmentations = torch.cat(segmentations)
                anomaly_maps = torch.cat(anomaly_maps)
                inputs = torch.cat(inputs)
                n_anomaly_scores = [x.numpy() for x, y in zip(anomaly_scores, labels) if y == 0]
                n_segmentations = torch.cat([x for x, y in zip(segmentations, labels) if y == 0])
                n_anomaly_maps = torch.cat([x for x, y in zip(anomaly_maps, labels) if y == 0])
                n_inputs = torch.cat([x for x, y in zip(inputs, labels) if y == 0])
                p_anomaly_scores = [x.numpy() for x, y in zip(anomaly_scores, labels) if y == 1]
                p_segmentations = torch.cat([x for x, y in zip(segmentations, labels) if y == 1])
                p_inputs = torch.cat([x for x, y in zip(inputs, labels) if y == 1])
                p_anomaly_maps = torch.cat([x for x, y in zip(anomaly_maps, labels) if y == 1])
                idx_hard = np.argpartition(-np.array(p_anomaly_scores), -10)[-10:]
                idx_easy = np.argpartition(np.array(p_anomaly_scores), -10)[-10:]
                print(np.array(p_anomaly_scores)[idx_hard])
                print(np.array(p_anomaly_scores)[idx_easy])
                # print(np.array(p_anomaly_scores))
                easy_maps = p_anomaly_maps[idx_easy].unsqueeze(1)
                easy_inp = p_inputs[idx_easy].unsqueeze(1)
                hard_maps = p_anomaly_maps[idx_hard].unsqueeze(1)
                hard_inp = p_inputs[idx_hard].unsqueeze(1)
                for i, _ in enumerate(hard_maps):
                    hard_maps[i, 0, 0, 0] == 1.
                    easy_maps[i, 0, 0, 0] == 1.
                # fp_maps = n_anomaly_maps[idx_n].unsqueeze(1)
                # fp_segs = n_segmentations[idx_n].unsqueeze(1)
                # print(tp_maps.shape)
                log({'anom_val/easy_maps': easy_maps,
                    'anom_val/easy_inputs': easy_inp,
                     'anom_val/hard_maps': hard_maps,
                     'anom_val/hard_inputs': hard_inp}, config)
                exit(0)
        if config.atlas_exp:
            if config.modality == 'MRI':
                #contrasts = torch.cat(contrasts)
                sizes = torch.cat(sizes)
                # mean_score_ratios = torch.cat(mean_score_ratios)
                # anomaly_scores = torch.cat(anomaly_scores)
                intensities = torch.cat(intensities)
                norm_intensities = torch.cat(norm_intensities)
                mean_intensities = torch.cat(mean_intensities)

                torch.save(
                    mean_intensities,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/mean_intensities_{config.sequence}_{config.brats_t1}_{config.method}.pt')

                torch.save(
                    norm_intensities,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/norm_intensities_{config.sequence}_{config.brats_t1}_{config.method}.pt')

                # torch.save(
                #     contrasts,
                #     f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/contrasts_{config.sequence}_{config.brats_t1}_{config.method}.pt')
                torch.save(
                    intensities,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/intensities_{config.sequence}_{config.brats_t1}_{config.method}.pt')
                torch.save(
                    sizes,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/sizes_{config.sequence}_{config.brats_t1}_{config.method}.pt')
                # torch.save(
                #     mean_score_ratios,
                #     f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/mean_score_ratios_{config.sequence}_{config.brats_t1}_{config.method}.pt')
                # torch.save(anomaly_scores,
                #            f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/anomaly_scores_{config.sequence}_{config.brats_t1}_{config.method}.pt')

            elif config.modality == 'RF':
                #contrasts = torch.cat(contrasts)
                sizes = torch.cat(sizes)
                # mean_score_ratios = torch.cat(mean_score_ratios)
                # anomaly_scores = torch.cat(anomaly_scores)
                intensities = torch.cat(intensities)
                norm_intensities = torch.cat(norm_intensities)
                mean_intensities = torch.cat(mean_intensities)

                torch.save(
                    mean_intensities,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/mean_intensities_RF_{config.method}.pt')

                torch.save(
                    norm_intensities,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/norm_intensities_RF_{config.method}.pt')

                # torch.save(
                #     contrasts,
                #     f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/contrasts_{config.sequence}_{config.brats_t1}_{config.method}.pt')
                torch.save(
                    intensities,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/intensities_RF_{config.method}.pt')
                torch.save(
                    sizes,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/sizes_RF_{config.method}.pt')
            else:
                #contrasts = torch.cat(contrasts)
                sizes = torch.cat(sizes)
                # mean_score_ratios = torch.cat(mean_score_ratios)
                # anomaly_scores = torch.cat(anomaly_scores)
                intensities_r = torch.cat(intensities_r)
                intensities_g = torch.cat(intensities_g)
                intensities_b = torch.cat(intensities_b)
                norm_intensities_r = torch.cat(norm_intensities_r)
                norm_intensities_g = torch.cat(norm_intensities_g)
                norm_intensities_b = torch.cat(norm_intensities_b)

                torch.save(
                    norm_intensities_r,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/norm_intensities_r_COL_{config.method}.pt')
                torch.save(
                    norm_intensities_g,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/norm_intensities_g_COL_{config.method}.pt')
                torch.save(
                    norm_intensities_b,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/norm_intensities_b_COL_{config.method}.pt')

                torch.save(
                    intensities_r,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/intensities_r_COL_{config.method}.pt')
                torch.save(
                    intensities_g,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/intensities_g_COL_{config.method}.pt')
                torch.save(
                    intensities_b,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/intensities_b_COL_{config.method}.pt')
                torch.save(
                    sizes,
                    f'/u/home/lagi/thesis/UAD_study/histogram stuff/statistics/sizes_COL_{config.method}.pt')

            exit(0)

        # calculate metrics like AP, AUROC, on pixel and/or image level

        if config.modality in ['CXR', 'OCT'] or (config.dataset != 'DDR' and config.modality == 'RF'):
            segmentations = None  # disables pixel level evaluation in metrics()

        threshold = metrics(config, anomaly_maps, segmentations, anomaly_scores, labels)

        # if config.normal_fpr:
        #     config.nfpr = 0.01
        #     dice_f1_normal_nfpr(val_loader, config, val_step,
        #                         anomaly_maps, segmentations, anomaly_scores, labels)
        #     config.nfpr = 0.05
        #     dice_f1_normal_nfpr(val_loader, config, val_step,
        #                         anomaly_maps, segmentations, anomaly_scores, labels)
        #     config.nfpr = 0.1
        #     dice_f1_normal_nfpr(val_loader, config, val_step, anomaly_maps,
        #                         segmentations, anomaly_scores, labels)

        # if config.patches:
        #     return

    # do a single forward pass to extract stuff to log
    # the batch size is num_images_log for test_loaders, so a single forward pass is enough
    input, mask = next(iter(test_loader))

    if config.patches:
        input = input.view(input.shape[0] * input.shape[1], *input.shape[2:])
    # forward pass, x = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
    x = val_step(input.to(config.device), return_loss=False)

    if config.patches:
        input = input.view(4, config.img_channels, config.image_size * int(math.sqrt(config.num_patches)),
                           config.image_size * int(math.sqrt(config.num_patches)))
        map = x[0].view(4, 1, config.image_size * int(math.sqrt(config.num_patches)),
                        config.image_size * int(math.sqrt(config.num_patches)))

        mask = mask.squeeze(1)

    else:
        maps = x[0]

    log({'anom_val/input images': input,
         'anom_val/targets': mask,
         'anom_val/anomaly maps': maps}, config)

    # # Log images to wandb
    # input_images = list(input[:config.num_images_log].cpu())1
    # targets = list(mask.float()[:config.num_images_log].cpu())

    # anomaly_images = list(maps[:config.num_images_log].cpu())
    # config.logger.log({
    #     'anom_val/input images': [wandb.Image(img) for img in input_images],
    #     'anom_val/targets': [wandb.Image(img) for img in targets],
    #     'anom_val/anomaly maps': [wandb.Image(img) for img in anomaly_images]
    # }, step=config.step)

    # if recon based method, len(x)==3 because val_step returns reconstructions.
    # if thats the case  log the reconstructions
    if len(x) == 3:

        log({'anom_val/reconstructions': x[2]}, config)
    # produce thresholded images
    # if not config.limited_metrics or segmentations is not None:
    #     # create thresholded images on the threshold of best posible dice score on the dataset
    #     anomaly_thresh_map = torch.where(x[0] < threshold,
    #                                      torch.FloatTensor([0.]).to(config.device),
    #                                      torch.FloatTensor([1.]).to(config.device))

    #     log({'anom_val/thresholded maps': anomaly_thresh_map}, config)


def evaluate_hist(config: Namespace, test_loader: DataLoader,
                  val_step: object, val_loader: DataLoader = None) -> None:
    """
    because we work we datasets of both with and without masks, code is structured in a
    way that no mask dataloaders also return masks. For healthy samples they return empty
    masks, and for anomalous samples they return non-empty masks. The code then works the
    same and the binary label is produced by label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
    """
    labels = []
    anomaly_scores = []
    anomaly_maps = []
    segmentations = []
    normal_recon_pixels = []
    abnormal_recon_pixels = []

    # forward pass the testloader to extract anomaly maps, scores, masks, labels
    benchmark_step = 0
    total_elapsed_time = 0
    for input, mask in tqdm(test_loader, desc="Test set"):

        # forward pass, x = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
        x = val_step(input.to(config.device), return_loss=False)
        if config.modality in ['MRI', 'MRInoram', 'CT']:
            brainmask = torch.stack([inp > inp.min() for inp in input])
            anomaly_map = x[0]
            anomaly_map = anomaly_map[brainmask > 0]
            mask_for_dice = mask[brainmask > 0]
            anomaly_maps.append(anomaly_map.cpu())
            segmentations.append(mask_for_dice)

        anomaly_scores.append(x[1].cpu())
        label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
        labels.append(label)

    # if config.normal_fpr:
    #     thresh_list = []
    #     nfpr_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    #     for i in nfpr_list:
    #         config.nfpr = i
    #         thresh_list.append(dice_f1_normal_nfpr(val_loader, config, val_step,
    #                                                anomaly_maps, None, anomaly_scores, labels))

    negatives = len([i for i in torch.cat(labels) if i == 0])
    scores = torch.cat(anomaly_scores)
    label_tensor = torch.cat(labels)

    fontsize_legends = 10
    # # pixel histogram
    # norm_recon_pixels = [item for sublist in normal_recon_pixels for item in sublist]
    # abnorm_recon_pixels = [item for sublist in abnormal_recon_pixels for item in sublist]

    # maxi = max(abnorm_recon_pixels)
    # plt.figure(0, figsize=(10, 5))
    # plt.hist(norm_recon_pixels, bins=100, label='normal pixels', range=(0, maxi), alpha=0.5)
    # plt.hist(abnorm_recon_pixels, bins=100, label='anomal pixels', range=(0, maxi), alpha=0.5)
    # plt.legend(loc='upper right', fontsize=fontsize_legends)
    # plt.xlabel("reconstruction error", fontsize=fontsize_legends)
    # plt.ylabel("frequency", fontsize=fontsize_legends)

    # plt.tight_layout()
    # plt.savefig('pixel-histogram.png')
    import numpy as np
    # image histogram
    maxi = torch.cat(anomaly_scores).max().numpy()
    test_normals = torch.cat(anomaly_scores)[
        torch.cat(labels) == 0].numpy()
    test_anomals = torch.cat(anomaly_scores)[
        torch.cat(labels) == 1].numpy()
    # y = torch.cat(anomaly_scores)[
    #     torch.cat(segmentations).sum(dim=(1, 2, 3)) > 0].numpy()
    anomaly_scores_val = []
    # segmentations = []
    for input in tqdm(val_loader, desc="Test set"):

        # forward pass, x = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
        x = val_step(input.to(config.device), return_loss=False)
        # if config.modality in ['MRI', 'MRInoram', 'CT']:
        #     brainmask = torch.stack([inp > inp.min() for inp in input])
        #     anomaly_map = x[0]
        #     anomaly_map = anomaly_map[brainmask > 0]
        #     mask_for_dice = mask[brainmask > 0]
        #     anomaly_maps.append(anomaly_map.cpu())

        anomaly_scores_val.append(x[1].cpu())
        # label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
        # labels.append(label)
    val_normals = torch.cat(anomaly_scores_val).numpy()
    mini = torch.cat(anomaly_scores).min().numpy()
    plt.figure(1, figsize=(10, 5))
    plt.hist(test_normals, bins=100, label='normal test samples',
             range=(mini, maxi), alpha=0.5, density=True)
    plt.hist(val_normals, bins=100, label='normal val samples',
             range=(mini, maxi), alpha=0.5, density=True)
    plt.hist(test_anomals, bins=100, label='anomal test samples',
             range=(mini, maxi), alpha=0.5, density=True)
    # t_0 = np.ones((500)) * t0
    # t_1 = np.ones((500)) * t1
    # t_2 = np.ones((500)) * t2
    # t_3 = np.ones((500)) * t3
    # t_4 = np.ones((500)) * t4
    # plt.hist(t_0, bins=700, label='1% FPR', range=(mini, maxi), alpha=1)
    # plt.hist(t_1, bins=700, label='5% FPR', range=(mini, maxi), alpha=1)
    # plt.hist(t_2, bins=700, label='10% FPR', range=(mini, maxi), alpha=1)
    # plt.hist(t_3, bins=700, label='20% FPR', range=(mini, maxi), alpha=1)
    # plt.hist(t_4, bins=700, label='30% FPR', range=(mini, maxi), alpha=1)

    plt.legend(loc='upper right', fontsize=fontsize_legends)
    plt.xlabel("anomaly score", fontsize=fontsize_legends)
    plt.ylabel("frequency", fontsize=fontsize_legends)

    plt.tight_layout()
    plt.savefig(f'image-histogram-{config.name}.png')
    # if config.modality in ['CXR', 'RF']:
    #     exit(0)
    # if config.normal_fpr:
    #     thresh_list = []
    #     nfpr_list = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    #     for i in nfpr_list:
    #         config.nfpr = i
    #         thresh_list.append(dice_f1_normal_nfpr(val_loader, config, val_step,
    #                                                anomaly_maps, segmentations, anomaly_scores, labels))

    # segmentations = np.asarray(torch.cat(segmentations)).reshape(-1)
    # anomaly_maps = np.asarray(torch.cat(anomaly_maps)).reshape(-1)
    # num_negatives = len(segmentations) - np.count_nonzero(segmentations)
    # negatives = anomaly_maps[segmentations == 0]

    # for i, nfpr in enumerate(nfpr_list):

    #     false_positives = negatives[negatives > thresh_list[i]]
    #     num_false_positives = len(false_positives)
    #     print(f'Pixel fpr with {nfpr * 100}%-nfpr:', num_false_positives / num_negatives)
    # calculate metrics like AP, AUROC, on pixel and/or image level

    # if config.modality in ['CXR', 'OCT'] or (config.dataset != 'IDRID' and config.modality == 'RF'):
    #     segmentations = None  # disables pixel level evaluation in metrics()

    # threshold = metrics(config, anomaly_maps, segmentations, anomaly_scores, labels)

    # if config.patches:
    #     return

    # labels = []
    # anomaly_scores = []
    # anomaly_maps = []
    # segmentations = []

    # anomaly_scores_l1 = []
    # anomaly_maps_l1 = []
    # anomaly_scores_l1_gb = []
    # anomaly_maps_l1_gb = []
    # anomaly_scores_gb_ssim = []
    # anomaly_maps_gb_ssim = []

    # # forward pass the testloader to extract anomaly maps, scores, masks, labels
    # for input, mask in tqdm(test_loader, desc="Test set"):

    #     # forward pass, x = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
    #     x = val_step(input.to(config.device), return_loss=False)

    #     anomaly_maps_l1.append(x[0].cpu())
    #     anomaly_scores_l1.append(x[1].cpu())
    #     anomaly_maps_l1_gb.append(x[2].cpu())
    #     anomaly_scores_l1_gb.append(x[3].cpu())
    #     anomaly_maps_gb_ssim.append(x[4].cpu())
    #     anomaly_scores_gb_ssim.append(x[5].cpu())
    #     segmentations.append(mask)
    #     label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
    #     labels.append(label)

    # # calculate metrics like AP, AUROC, on pixel and/or image level

    # if config.modality in ['CXR', 'OCT'] or (config.dataset != 'IDRID' and config.modality == 'RF'):
    #     segmentations = None  # disables pixel level evaluation in metrics()
    # print("l1_gb")
    # threshold = metrics(config, anomaly_maps_l1_gb, segmentations, anomaly_scores_l1_gb, labels)
    # print("gb_ssim")
    # threshold = metrics(config, anomaly_maps_gb_ssim, segmentations, anomaly_scores_gb_ssim, labels)
    # print("l1")
    # threshold = metrics(config, anomaly_maps_l1, segmentations, anomaly_scores_l1, labels)
    # exit(0)


#    if config.patches:
#                 # input.shape = [samples,crops,c,h,w]
#                 # mask.shape = [samples,crops,1,h,w]
#                 b, num_crops, _, _, _ = input.shape
#                 input = input.view(input.shape[0] * input.shape[1], *input.shape[2:])  # [samples*crops,c,h,w]
#                 #  print(input.shape)
#                 # mask = mask.reshape(mask.shape[0] * mask.shape[1], *mask.shape[2:])# [samples*crops,1,h,w]
#                 # print(mask.shape)
#                 x = val_step(input.to(config.device), return_loss=False)
#                 # print(x[1])
#                 per_crop_scores = x[1].view(b, config.num_patches)  # [samples,crops]
#                 # print(per_crop_scores.shape)
#                 import math

#                 anomaly_score = per_crop_scores.mean(1)  # [samples]
#                 discard_nan = False
#                 for i in anomaly_score:

#                     if math.isnan(i):
#                         discard_nan = True
#                 if not discard_nan:
#                     # print(anomaly_score.shape)
#                     anomaly_scores.append(anomaly_score.cpu())
#                     # [samples, crops] crop-wise binary vector
#                     crop_label = torch.where(mask.sum(dim=(2, 3, 4)) > 0, 1, 0)  # [samples, crops]
#                     # print(crop_label.shape)
#                     # [samples] sample-wise binary vector
#                     label = torch.where(crop_label.sum(dim=1) > 0, 1, 0)
#                     # print(label.shape)
#                     labels.append(label)
#                     if config.dataset == 'IDRID':

#                         map = x[0].view(b, 1, config.image_size * int(math.sqrt(config.num_patches)),
#                                         config.image_size * int(math.sqrt(config.num_patches)))
#                         anomaly_maps.append(map.cpu())
#                         segmentations.append(mask)
