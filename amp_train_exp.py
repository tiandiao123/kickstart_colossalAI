import os
from pathlib import Path
import logging

import torch
from timm.models import vit_base_patch16_224
from titans.utils import barrier_context
from torchvision import datasets, transforms

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR

# 初始化分布式设置
parser = colossalai.get_default_parser()
args = parser.parse_args()

# launch from torch
colossalai.launch_from_torch(config=dict())

# define the constants
NUM_EPOCHS = 2
BATCH_SIZE = 128
# build model
model = vit_base_patch16_224(drop_rate=0.1)

# build dataloader
train_dataset = datasets.Caltech101(
    root=Path(os.environ['DATA']),
    download=True,
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
    ]))

# build optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=0.1)

# build loss
criterion = torch.nn.CrossEntropyLoss()

# lr_scheduelr
lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=50, total_steps=NUM_EPOCHS)

plugin = TorchDDPPlugin()
train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
booster = Booster(mixed_precision='fp16', plugin=plugin)

# if you need to customize the config, do like this
# >>> from colossalai.mixed_precision import FP16TorchMixedPrecision
# >>> mixed_precision = FP16TorchMixedPrecision(
# >>>     init_scale=2.**16,
# >>>     growth_factor=2.0,
# >>>     backoff_factor=0.5,
# >>>     growth_interval=2000)
# >>> plugin = TorchDDPPlugin()
# >>> booster = Booster(mixed_precision=mixed_precision, plugin=plugin)

# boost model, optimizer, criterion, dataloader, lr_scheduler
model, optimizer, criterion, dataloader, lr_scheduler = booster.boost(model, optimizer, criterion, train_dataloader, lr_scheduler)


model.train()
for epoch in range(NUM_EPOCHS):
    for img, label in enumerate(train_dataloader):
        logging.warning(img.shape)
        img = img.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        booster.backward(loss, optimizer)
        optimizer.step()
    lr_scheduler.step()
