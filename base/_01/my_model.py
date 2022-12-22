import numpy as np

import torch
import copy

from torch import nn

class CLIPComplete(nn.Module):
    def __init__(self, model_visual):
        super().__init__()
        self.v1 = copy.deepcopy(model_visual)
        self.v2 = copy.deepcopy(model_visual)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    @property
    def dtype(self):
        return self.v1.conv1.weight.dtype
        
    def encode_image(self, image, model):
        return model(image.type(self.dtype))
    
    def forward(self, img1, img2):
        img1_features = self.encode_image(img1, self.v1)
        img2_features = self.encode_image(img2, self.v2)

        # normalized features
        img1_features = img1_features / img1_features.norm(dim=1, keepdim=True)
        img2_features = img2_features / img2_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_img1 = logit_scale * img1_features @ img2_features.t()
        logits_per_img2 = logits_per_img1.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_img1, logits_per_img2