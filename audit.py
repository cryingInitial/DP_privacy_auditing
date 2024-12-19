import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import Models
from utils.data import load_data
from utils.audit import compute_eps_lower_from_mia
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from copy import deepcopy

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MIA')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset')
    parser.add_argument('--target', type=str, default='clipbkd', help='Target')
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4,5,6,7', help='Seeds')
    parser.add_argument('--eps', type=float, default=4.0, help='Eps')
    parser.add_argument('--methods', type=str, default='subtract,ours', help='Methods')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--surrogate', type=bool, default=True, help='Surrogate')
    parser.add_argument('--lr', type=float, default=0.1, help='LR')
    return parser.parse_args()

def get_attack_results(dataset='mnist', target='blank', seeds=[0,1,2,3,4,5,6,7], eps=4.0, methods=['subtract', 'ours'], epochs=10, surrogate=True, lr=0.1):
    full_or_worst = 'worst' if worst else 'full'
    internal = 'black' if surrogate else 'white'
    base_eps, subtract_eps, ours_eps = 0, 0, 0

    os.makedirs(f'results_{full_or_worst}_{alpha}/{dataset}_{target}/{internal}_{eps}', exist_ok=True)
    with open(f'results_{full_or_worst}_{alpha}/{dataset}_{target}/{internal}_{eps}/{epochs}.txt', 'a') as f:
        f.write(f'Epochs: {epochs}, Eps: {eps}, Alpha: {alpha}, dynamicY: {dynamicY}, dataset: {dataset}, target: {target} LR: {lr}\n')
    # PREPARE DIRECTORIES
    directories = []
    for seed in seeds:
        if full_or_worst == 'worst':
            directories.append(f'exp_data_{full_or_worst}/{dataset}_{target}_100/seed{seed}/{dataset}_half_cnn_eps{eps}')
        else:
            directories.append(f'exp_data_{full_or_worst}/{dataset}_{target}_100/seed{seed}/{dataset}_cnn_eps{eps}')

    combs = []
    for seed in seeds: combs.append([seed])

    for idx, comb in enumerate(combs):
        models_in, models_out, surrogate_models_in, surrogate_models_out = load_models(directories, comb, dataset)
        criterion = nn.CrossEntropyLoss()
        base_eps = get_eps_from_loss(models_in, models_out, dataset, criterion, path=f'results_full_{alpha}/{dataset}_{target}/{internal}_{eps}/base_{epochs}_{idx}.pdf')
        subtract_eps = base_eps
        ours_eps = base_eps

        for method in methods:
            
            fixed_input = torch.randn(1, 1, 28, 28).to('cuda') if dataset == 'mnist' else torch.randn(1, 3, 32, 32).to('cuda')
            fixed_output = torch.tensor([9]).to('cuda')
            print(f'Fixed output: {target}')

            X = torch.nn.Parameter(torch.zeros(1, 1, 28, 28).to('cuda'), requires_grad=True) if dataset == 'mnist' else torch.nn.Parameter(torch.ones(1, 3, 32, 32).to('cuda'), requires_grad=True)
            optimizer_input = optim.Adam([X], lr=lr)
            y = torch.tensor([9]).to('cuda')

            if surrogate:
                models_train_in = surrogate_models_in
                models_train_out = surrogate_models_out
                print("SURROGATE")
            else:
                models_train_in = models_in
                models_train_out = models_out
                print(len(models_train_in), len(models_train_out))
                print("ORIGINAL")

            for epoch in tqdm(range(epochs)):
                optimizer_input.zero_grad()
                
                in_outputs = []; out_outputs = []
                for model_in, model_out in zip(models_train_in, models_train_out):
                    
                    in_output = model_in(X.to('cuda'))
                    out_output = model_out(X.to('cuda'))

                    in_outputs.append(in_output)
                    out_outputs.append(out_output)

                losses = []
                losses_ins = []
                losses_outs = []
                mean_of_out_outputs_loss = sum([criterion(out_output, y) for out_output in out_outputs]) / len(out_outputs)
                for in_output, out_output in zip(in_outputs, out_outputs):
                    
                    if method == 'subtract':
                        loss = criterion(in_output, y) - criterion(out_output, y)

                    elif method == 'ours':
                        loss = criterion(in_output, y) - mean_of_out_outputs_loss + alpha
                        loss = torch.clamp(loss, min=0)
                        losses_ins.append(criterion(in_output, y))
                        losses_outs.append(criterion(out_output, y))

                    losses.append(loss)

                loss = sum(losses)
                loss.backward()
                optimizer_input.step()
                X.data.clamp_(0, 1)
            
                print(f'epoch: {epoch}, loss: {loss.item()}')

                inputX = X.detach()
                inputy = None

                if surrogate:
                    if method == 'subtract': 
                        renew_eps = get_eps_from_loss(models_in, models_out, dataset, criterion, inputX=inputX, inputy=inputy, path=f'results_full_{alpha}/{dataset}_{target}/{internal}_{eps}/subtract_{epochs}_{idx}.pdf')
                        if renew_eps > subtract_eps:
                            print(f'New eps: {renew_eps}, Old eps: {subtract_eps}')
                            subtract_eps = renew_eps
                    elif method == 'ours':
                        renew_eps = get_eps_from_loss(models_in, models_out, dataset, criterion, inputX=inputX, inputy=inputy, path=f'results_full_{alpha}/{dataset}_{target}/{internal}_{eps}/ours_{epochs}_{idx}.pdf')
                        if renew_eps > ours_eps:
                            print(f'New eps: {renew_eps}, Old eps: {ours_eps}')
                            ours_eps = renew_eps
            if not surrogate:
                if method == 'subtract': subtract_eps = get_eps_from_loss(models_in, models_out, dataset, criterion, inputX=inputX, inputy=inputy, path=f'results_full_{alpha}/{dataset}_{target}/{internal}_{eps}/subtract_{epochs}_{idx}.pdf')
                elif method == 'ours': ours_eps = get_eps_from_loss(models_in, models_out, dataset, criterion, inputX=inputX, inputy=inputy, path=f'results_full_{alpha}/{dataset}_{target}/{internal}_{eps}/ours_{epochs}_{idx}.pdf')
                elif method == 'KL': ours_eps = get_eps_from_loss(models_in, models_out, dataset, criterion, inputX=inputX, inputy=inputy, path=f'results_full_{alpha}/{dataset}_{target}/{internal}_{eps}/KL_{epochs}_{idx}.pdf')

        os.makedirs(f'results_{full_or_worst}_{alpha}/{dataset}_{target}/{internal}_{eps}', exist_ok=True)
        with open(f'results_{full_or_worst}_{alpha}/{dataset}_{target}/{internal}_{eps}/{epochs}.txt', 'a') as f:
            f.write(f'Base eps: {base_eps}, Subtract_eps: {subtract_eps}, Ours_eps: {ours_eps}\n')


def get_eps_from_loss(models_in, models_out, dataset, criterion, inputX=None, inputy=None, path=None):

    in_outputs = []
    out_outputs = []

    print(f'inputX: {True if inputX is not None else False}, inputy: {True if inputy is not None else False}')
    for model_in, model_out in zip(models_in, models_out):
        inpX = inputX if inputX is not None else torch.zeros(1, 1, 28, 28).cuda() if dataset == 'mnist' else torch.zeros(1, 3, 32, 32).cuda()
        inpY = inputy if inputy is not None else torch.tensor([9]).cuda()
        # print(f"대반전 {inpY}") 
        
        if inputX is not None: inpX = torch.clamp(inpX, min=0, max=1)
        if inputy is not None: inpY = torch.nn.functional.softmax(inpY, dim=1)

        in_output = model_in(inpX)
        out_output = model_out(inpX)

        in_target = deepcopy(inpY)
        out_target = deepcopy(inpY)

        in_outputs.append(-criterion(in_output, in_target).item())
        out_outputs.append(-criterion(out_output, out_target).item())

    mia_scores = np.concatenate([in_outputs, out_outputs])
    mia_labels = np.concatenate([np.ones_like(in_outputs), np.zeros_like(out_outputs)])
    _, emp_eps_loss = compute_eps_lower_from_mia(mia_scores, mia_labels, 0.05, 1e-5, 'GDP', n_procs=1)

    in_outputs = - np.array(in_outputs)
    out_outputs = - np.array(out_outputs)
    plt.figure()
    plt.hist(in_outputs, bins=20, alpha=0.5, label='in')
    plt.hist(out_outputs, bins=20, alpha=0.5, label='out')
    plt.legend()

    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    if path: plt.savefig(path)

    return emp_eps_loss

def load_models(directories, comb, dataset):
    models_in = []
    models_out = []
    surrogate_models_in = []
    surrogate_models_out = []
    directories = [directories[i] for i in comb]
    for directory in directories:
        for model_loc in os.listdir(f'{directory}/models'):
            if dataset == 'mnist': model = Models['cnn'](in_shape=(1, 1, 28, 28), out_dim=10).to('cuda')
            elif dataset == 'cifar10': model = Models['cnn'](in_shape=(1, 3, 32, 32), out_dim=10).to('cuda')

            model.load_state_dict(torch.load(os.path.join(f'{directory}/models', model_loc)))
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            if 'attack' in model_loc:
                if 'in' in model_loc: surrogate_models_in.append(model)
                elif 'out' in model_loc: surrogate_models_out.append(model)

    print(f'Loaded {len(models_in)} in models and {len(models_out)} out models')
    print(f'Loaded {len(surrogate_models_in)} surrogate in models and {len(surrogate_models_out)} surrogate out models')

    return models_in, models_out, surrogate_models_in, surrogate_models_out

if __name__ == '__main__':
    args = parse_args()
    get_attack_results(dataset=args.dataset, target=args.target, eps=args.eps, epochs=args.epochs, surrogate=False, lr=args.lr)
