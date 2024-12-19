import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.cnn import CNN
from utils.data import load_data
from tqdm import tqdm
import argparse
import torchvision.models as models


import os
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model', type=str, default='cnn')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--canary', type=str, default='blank')
parser.add_argument('--epsilon', type=float, default=10.0)
args = parser.parse_args()

path = f"exp_data_full/{args.dataset}_{args.canary}_100/seed{args.seed}/{args.dataset}_cnn_eps{args.epsilon}"

X, y, outdim = load_data(args.dataset, None, split='train')

X_target = np.load(f'{path}/target_X.npy')
y_target = np.load(f'{path}/target_y.npy')

print(X_target, y_target)
X = torch.cat([X, torch.tensor(X_target).float()])
y = torch.cat([y, torch.tensor(y_target)])


lists = os.listdir(f'{path}/models')
lists = [pth for pth in lists if 'attack' not in pth]
print("Total models: ", len(lists))

for pth in tqdm(lists):

    original_model = CNN(in_shape=X.shape, out_dim=outdim, dropout_rate=0.0).cuda()
    if args.model == 'cnn': attack_model = CNN(in_shape=X.shape, out_dim=outdim, dropout_rate=0.0).cuda()
    elif args.model == 'resnet18': 
        attack_model = models.resnet18(num_classes=outdim).cuda()
        if args.dataset == 'mnist':
            attack_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=3, bias=False).cuda()
        elif args.dataset == 'cifar10':
            attack_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).cuda()

    original_model.load_state_dict(torch.load(f"{path}/models/{pth}"))

    for param in original_model.parameters():
        param.requires_grad = False
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    temperature = 1.0
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-4)

    # distillation
    for epoch in tqdm(range(200)):
        losses = []
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            original_output = original_model(images)
            conjugate_output = attack_model(images)

            loss = criterion(F.log_softmax(conjugate_output / temperature, dim=1),
                             F.softmax(original_output / temperature, dim=1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        
        print(f'epoch: {epoch}, loss: {sum(losses) / len(losses)}')

    # check accuracy
    X_test, y_test, _ = load_data(args.dataset, None, split='test')
    correct = 0
    total = 0

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    attack_model.eval()
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        output = attack_model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'accuracy: {correct / total}')
    torch.save(attack_model.state_dict(), f"{path}/models/{pth.split('.')[0]}_attack_{args.model}.pt")
