import torch
from torch import nn
from models.resnet import WaveNet,ResNet18
import tensorwatch
from torchvision.models import resnet18
models=ResNet18()
sd=tensorwatch.draw_model(models,[32,3,32,32])
print(sd)