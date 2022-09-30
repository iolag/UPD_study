
"""
adapted from https://github.com/YoungGod/DFR and modified to work with efficient PyTorch modules
"""

import torch


def estimate_latent_channels(extractor, train_loader):
    """
    Estimate the number of latent channels for the Feature Autoencoder
    by performing a PCA over the features extracted from the train set.
    """
    device = next(extractor.parameters()).device
    feats = []
    i_samples = 0
    for i, normal_img in enumerate(train_loader):
        # Extract features
        with torch.no_grad():
            feat = extractor(normal_img.to(device))  # b, c, h, w
        # Reshape
        b, c = feat.shape[:2]
        feat = feat.permute(0, 2, 3, 1).reshape(-1, c)  # b*h*w, c
        feats.append(feat)
        del feat
        i_samples += b
        if i_samples > 20:
            break

    # Concatenate and center feats
    feats = torch.cat(feats, axis=0)

    mean = torch.mean(feats, dim=0)
    feats -= mean

    n_samples = feats.shape[0]
    s = torch.linalg.svdvals(feats.cpu())

    # transform sing. values to eigenvalues (=explained_variance)
    explained_variance = s ** 2 / (n_samples - 1)

    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance

    cumulative_explained_var_ratio = list(torch.cumsum(explained_variance_ratio, 0))
    latent_channels = len([i for i in cumulative_explained_var_ratio if i <= 0.9])
    del feats
    return latent_channels
