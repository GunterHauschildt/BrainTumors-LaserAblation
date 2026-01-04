from monai.transforms import MapTransform
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


class BuildRewardFieldD(MapTransform):
    def __init__(self, keys=("labels",), reward_key="reward",
                 tumor_id=10, healthy_ids=range(1,10)):
        super().__init__(keys)
        self.reward_key = reward_key
        self.tumor_id = tumor_id
        self.healthy_ids = list(healthy_ids)

    def _dist_to_potential(self, D, sigma):
        return torch.exp(-D / sigma)

    def __call__(self, data):
        d = dict(data)

        lbl = d[self.keys[0]].cpu().numpy().astype(np.int16)  # (R,A,S)

        C = int(lbl.max()) + 1
        dist_fields = []

        for c in range(C):
            mask = (lbl == c)
            dist = distance_transform_edt(~mask)
            dist_fields.append(torch.from_numpy(dist).float())

        D = torch.stack(dist_fields, dim=0)  # (C,R,A,S)

        tumor_field = self._dist_to_potential(D[self.tumor_id], sigma=3.0)

        healthy_field = torch.zeros_like(tumor_field)
        for i in self.healthy_ids:
            healthy_field += self._dist_to_potential(D[i], sigma=2.5)

        bg_field = self._dist_to_potential(D[0], sigma=4.0)

        tumor_field   /= tumor_field.max()   + 1e-8
        healthy_field /= healthy_field.max() + 1e-8
        bg_field      /= bg_field.max()      + 1e-8

        reward = (
            +5.0 * tumor_field
            -1.0 * healthy_field
            -0.2 * bg_field
        )

        d[self.reward_key] = reward
        return d
