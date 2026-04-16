import torch.nn as nn


def init_weights(m, init_type='kaiming'):
    if isinstance(m, nn.Conv2d):
        if init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        elif init_type == 'normal':
            nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        if init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        elif init_type == 'normal':
            nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
