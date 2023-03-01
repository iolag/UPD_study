"""
adapted from: https://github.com/LilitYolyan/CutPaste
"""

import torch
import os
from tqdm import tqdm
import numpy as np
import pickle
import torch.nn.functional as F


class Localization:
    def __init__(self, model, trainloader, config, kernel_dim=(16, 16), stride=4):
        self.model = model
        del model
        self.model.eval()
        self.trainloader = trainloader
        self.kernel_dim = kernel_dim
        self.stride = stride
        self.fit_gde = config.fit_gde
        self.batch_size = config.batch_size
        self.device = config.device
        self.align = config.align

        # create save path for normal GDE statistics
        self.save_path = f'saved_statistics/{config.modality}'
        if config.modality == 'MRI':
            self.save_path += f'_{config.sequence}'
        self.save_path += f'_{config.seed}_GDE_patch.sav'
        # create folder for saved stats
        os.makedirs('saved_statistics', exist_ok=True)

    def extract_patch_embeddings(self, image):
        b, c, w, h = image.shape
        num_patches_each_dim = ((w - self.kernel_dim[0]) // self.stride) + 1
        patches = torch.nn.functional.unfold(image, self.kernel_dim, stride=self.stride)
        patches = patches.view(b, c, self.kernel_dim[0], self.kernel_dim[1], -1)
        patches = patches.permute(0, 4, 1, 2, 3)
        patches = patches.reshape(-1, c, self.kernel_dim[0], self.kernel_dim[1])

        with torch.no_grad():
            _, patch_embeddings = self.model(patches)

        patch_embeddings = patch_embeddings.reshape(b, num_patches_each_dim, num_patches_each_dim, -1)
        return patch_embeddings

    def patch_GDE_fit(self):
        if not os.path.exists(self.save_path) or self.fit_gde:
            embeds = []
            for img in tqdm(self.trainloader):
                if img.shape[1] == 1:
                    img = img.repeat(1, 3, 1, 1)
                img = img.to(self.device)
                patch_embeddings = self.extract_patch_embeddings(img)  # b x num_patches x num_patches
                b, w, h, c = patch_embeddings.shape
                if self.align:
                    patch_embeddings = patch_embeddings.reshape(b, w * h, c).cpu()
                else:
                    patch_embeddings = patch_embeddings.reshape(b * w * h, c).cpu()
                embeds.append(patch_embeddings)

            embeds = torch.cat(embeds)
            self.mean = torch.mean(embeds, axis=0)
            if self.align:
                ident = np.identity(embeds.shape[2])
                self.inv_cov = torch.zeros((c, c, w * h))
                for i in range(w * h):
                    cov = np.cov(embeds[:, i, :].numpy(), rowvar=False) + 0.01 * ident
                    self.inv_cov[:, :, i] = torch.from_numpy(np.linalg.inv(cov)).float()
            else:
                ident = np.identity(embeds.shape[1])
                cov = np.cov(embeds.numpy(), rowvar=False) + 0.01 * ident
                self.inv_cov = torch.from_numpy(np.linalg.inv(cov)).float()
            pickle.dump([self.mean, self.inv_cov], open(self.save_path, 'wb'))

        else:
            with open(self.save_path, "rb") as f:
                self.mean, self.inv_cov = pickle.load(f)
                self.inv_cov = self.inv_cov

    def patch_scores(self, input):
        patch_embeddings = self.extract_patch_embeddings(input)  # [14, 29, 29, 512]

        b, w, h, c = patch_embeddings.shape
        if self.align:
            patch_embeddings = patch_embeddings.reshape(b, w * h, c).cpu()
            distances = torch.zeros((b, w * h))
            for i in range(w * h):
                distances[:, i] = self.mahalanobis_distance(
                    patch_embeddings[:, i], self.mean[i, :], self.inv_cov[:, :, i])

        else:

            patch_embeddings = patch_embeddings.reshape(b * w * h, c).cpu()
            distances = self.mahalanobis_distance(patch_embeddings, self.mean, self.inv_cov)

        return distances.reshape(b, w, h)

    def anomaly_map(self, input):

        sp = self.patch_scores(input)
        up = F.interpolate(sp.unsqueeze(1), size=input.shape[-1],
                           mode='bilinear', align_corners=False).to(self.device)

        return up

    @staticmethod
    def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.
        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.
        from github.com/ORippler/gaussian-ad-mvtec/
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]
        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)

        return torch.where(dist < 0, 0, dist).sqrt()
