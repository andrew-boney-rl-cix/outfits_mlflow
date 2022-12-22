import numpy as np

import torch
import copy

from torch import nn

class CLIPComplete(nn.Module):
    def __init__(self, model_visual):
        super().__init__()
        self.visual = model_visual
        self.visual.proj = None # remove {output_dim} projection

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # add seperate projections for img1 and img2 
        # width can be found in https://arxiv.org/pdf/2103.00020.pdf
        width, output_dim = 768, model_visual.output_dim
        self.visual.proj1 = nn.Parameter(width ** -0.5 * torch.randn(width, output_dim))
        self.visual.proj2 = nn.Parameter(width ** -0.5 * torch.randn(width, output_dim))

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
        
    def encode_image(self, image, proj):
        return self.visual(image.type(self.dtype)) @ proj.type(self.dtype)

    def forward(self, img1, img2):
        img1_features = self.encode_image(img1, self.visual.proj1)
        img2_features = self.encode_image(img2, self.visual.proj2)
        
        # normalized features
        img1_features = img1_features / img1_features.norm(dim=1, keepdim=True)
        img2_features = img2_features / img2_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_img1 = logit_scale * img1_features @ img2_features.t()
        logits_per_img2 = logits_per_img1.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_img1, logits_per_img2