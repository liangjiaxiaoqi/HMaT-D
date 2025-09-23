import torch.nn as nn
from .t2t_vit import T2t_vit_t_14
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder


class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()

        # HMaT-D Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)
        self.depth_backbone = T2t_vit_t_14(pretrained=True, args=args)

        # HMaT-D Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # HMaT-D Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=args.img_size)

    def forward(self, image_Input, depth_Input):
        B, _, _, _ = image_Input.shape

        # HMaT-D Encoder
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_Input)
        depth_fea_1_16, _, _ = self.depth_backbone(depth_Input)

        # HMaT-D Convertor
        rgb_fea_1_16, depth_fea_1_16 = self.transformer(rgb_fea_1_16, depth_fea_1_16)

        # HMaT-D Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16, depth_fea_1_16)
        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4)

        return outputs
