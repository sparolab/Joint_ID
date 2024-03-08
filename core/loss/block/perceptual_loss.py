import torch
import torch.nn.functional as F
from torchvision import models
from ..loss_builder import LOSS_BLOCK


# --- Perceptual loss network  --- #
@LOSS_BLOCK.register_module()
class PerceptualLossNetwork(torch.nn.Module):
    def __init__(self, lambda_perceptual):
        super(PerceptualLossNetwork, self).__init__()
        self.vgg_layers = models.vgg19(pretrained=True)
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }
        
        self.lambda_perceptual = lambda_perceptual

    def output_features(self, x):
        output = {}
        self.vgg_layers.to(x.device)
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        final_loss = sum(loss)/len(loss)
        return self.lambda_perceptual * final_loss
    