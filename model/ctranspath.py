from timm.models.layers.helpers import to_2tuple
import timm
import torch
import torch.nn as nn


class ConvStem(nn.Module):# Come from: https://github.com/Xiyue-Wang/TransPath

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    # def freeze_backbone(self):
    #     linear_keyword = 'head'
    #     for name, param in self.named_parameters():
    #             if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
    #                 param.requires_grad = False#冻结此部分，不进行梯度更新。

    # def unfreeze_backbone(self):
    #     linear_keyword = 'head'
    #     for name, param in self.named_parameters():
    #             if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
    #                 param.requires_grad = True#解冻此部分，不进行梯度更新。

def ctranspath(input_shape=[224, 224], pretrained=False, num_classes=1000):
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    if pretrained:
        model.load_state_dict(torch.load("model_data/ctranspath.pth"),strict = False)
    if num_classes!=1000:
        in_features = model.head.in_features
        model.head = nn.Linear(model.num_features, num_classes)
    return model