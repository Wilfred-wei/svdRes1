import torch
import torch.nn as nn
from models.network.clip import clip


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class net_stage1(nn.Module):
    def __init__(self, dim=768, drop_rate=0.5, output_dim=1):
        super(net_stage1, self).__init__()

        # lode the frozen CLIP-ViT with wsgm trainable
        self.backbone, _ = clip.load('ViT-L/14', device='cpu')
        params = []
        for name, p in self.backbone.named_parameters():
            if ("WSGM" in name and "visual" in name) or name == "fc.weight" or name == "fc.bias":
                params.append(name)
            else:
                p.requires_grad = False
        # print(params)

        self.ln_post = LayerNorm(dim)

        self.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(dim, output_dim)
        )

    def forward(self, x, return_rcs=False, return_orth_loss: bool = False):
        if return_rcs:
            feature, cls_tokens, rcs_token = self.backbone.encode_image(x, return_rcs=True)
            result = self.fc(feature)
            return result, cls_tokens, rcs_token
        else:
            if return_orth_loss:
                feature, cls_tokens, orth_loss = self.backbone.encode_image(x, return_orth_loss=True)
                result = self.fc(feature)
                return result, cls_tokens, orth_loss
            else:
                feature, cls_tokens = self.backbone.encode_image(x, return_rcs=False)
                result = self.fc(feature)
                return result, cls_tokens


