import torch
import torch.nn.functional as F
from torch import nn

from .layer import GradientReversal


class FCOSDiscriminator(nn.Module):
    def __init__(self, num_convs=2, in_channels=256, grad_reverse_lambda=-1.0, grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator, self).__init__()

        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain


    def forward(self, feature, target, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        if self.grl_applied_domain == 'both':
            feature = self.grad_reverse(feature)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = self.grad_reverse(feature)
        x = self.dis_tower(feature)
        x = self.cls_logits(x)

        target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
        loss = self.loss_fn(x, target)

        return loss
