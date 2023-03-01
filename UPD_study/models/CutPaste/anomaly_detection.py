"""
adapted from: https://github.com/LilitYolyan/CutPaste
"""
import torch
import os
from tqdm import tqdm
import numpy as np
import pickle


class Detection:
    def __init__(self, model, trainloader, config):
        '''
        Anomaly Detection

        args:
        weights[str] _ path to weights
        device[str] _ device on wich model should be run
        '''
        self.model = model
        model.eval()
        self.trainloader = trainloader
        self.device = config.device
        self.fit_gde = config.fit_gde

        # create save path for normal GDE statistics
        self.save_path = f'saved_statistics/{config.modality}'
        if config.modality == 'MRI':
            self.save_path += f'_{config.sequence}'
        self.save_path += f'_{config.seed}_GDE.sav'
        # create folder for saved stats
        os.makedirs('saved_statistics', exist_ok=True)

    def create_embeds_train(self):

        embeddings = []

        with torch.no_grad():
            for imgs in tqdm(self.trainloader):
                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                _, embeds = self.model(imgs.to(self.device))
                embeddings.append(embeds.to('cpu'))
        return torch.cat(embeddings)

    def GDE_fit(self):
        if not os.path.exists(self.save_path) or self.fit_gde:
            train_embeds = self.create_embeds_train()
            self.mean = torch.mean(train_embeds, axis=0)
            ident = np.identity(train_embeds.shape[1])
            cov = np.cov(train_embeds.numpy(), rowvar=False) + 0.01 * ident
            self.inv_cov = torch.from_numpy(np.linalg.inv(cov)).float()
            # self.inv_cov = torch.Tensor(LedoitWolf().fit(train_embeds.cpu()).precision_,device="cpu")
            pickle.dump([self.mean, self.inv_cov], open(self.save_path, 'wb'))

        else:
            with open(self.save_path, "rb") as f:
                self.mean, self.inv_cov = pickle.load(f)
                self.inv_cov = self.inv_cov

    def GDE_scores(self, input):
        _, test_embeds = self.model(input)
        distances = self.mahalanobis_distance(test_embeds.cpu(), self.mean, self.inv_cov)
        return distances

    @staticmethod
    def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.
        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.
        from https://github.com/ORippler/gaussian-ad-mvtec/
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
        return dist.sqrt()
