import os 
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet34
from tqdm import tqdm

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.logging import get_dist_logger
from colossalai.nn.lr_scheduler import CosineAnnealingLR


colossalai.launch_from_torch(config=dict())
logger = get_dist_logger()

NUM_EPOCHS = 200
BATCH_SIZE = 128
GRADIENT_CLIPPING = 0.1

model = resnet34(num_classes=10)

# build dataloaders
train_dataset = CIFAR10(root=Path(os.environ.get('DATA', './data')),
                        download=True,
                        transform=transforms.Compose([
                            transforms.RandomCrop(size=32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                        ]))
# build criterion
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# lr_scheduler
lr_scheduler = CosineAnnealingLR(optimizer, total_steps=NUM_EPOCHS)


plugin = TorchDDPPlugin()
booster = Booster(plugin=plugin)
train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
model, optimizer, criterion, train_dataloader, lr_scheduler = booster.boost(model, optimizer, criterion, train_dataloader, lr_scheduler)


# verify gradient clipping
model.train()
for idx, (img, label) in enumerate(train_dataloader):
    img = img.cuda()
    label = label.cuda()

    model.zero_grad()
    output = model(img)
    train_loss = criterion(output, label)
    booster.backward(train_loss, optimizer)
    optimizer.clip_grad_by_norm(max_norm=GRADIENT_CLIPPING)
    optimizer.step()
    lr_scheduler.step()

    ele_1st = next(model.parameters()).flatten()[0]
    logger.info(f'iteration {idx}, loss: {train_loss}, 1st element of parameters: {ele_1st.item()}')

    # only run for 4 iterations
    if idx == 3:
        break

