import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cifar10_LT import make_cifar10_lt
from collections import Counter
from model import ResNet, Bottleneck, MLP

import os
import argparse
from tqdm import tqdm


def load_cifar10(args):
    train_data = make_cifar10_lt(
        root = '/home/suno3534/data/_datasets/',
        imb_ratio = args.imb_ratio, 
        is_train = True, 
        download = True, 
    )

    test_data = make_cifar10_lt(
        root = '/home/suno3534/data/_datasets/',
        imb_ratio = args.imb_ratio, 
        is_train = False, 
        download = True, 
    )

    print("Train 샘플 수:", len(train_data))
    print("Test 샘플 수 :", len(test_data))

    # 각 클래스별 개수 확인
    imgs, labels = zip(*train_data)
    cls_counts = Counter(labels)
    print("클래스별 샘플 수:", cls_counts)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

def calcul_feature_mean(z, labels, num_class):
    label_list = list(range(num_class))

    feature_mean = torch.stack([z[labels == label].mean(dim=0) for label in label_list], dim=0)
    nan_mask = torch.isnan(feature_mean).any(dim=1)  # 각 행에 NaN이 있는지 확인
    feature_mean[nan_mask] = 0.0

    return feature_mean

def center_and_normalize(mu, z_all):
    mu_g = z_all.mean(dim=0, keepdim=True)
    mu_tilde = mu - mu_g
    mu_tilde = F.normalize(mu_tilde, p=2, dim=1)
    
    return mu_tilde

def P2P(matrix, c):
    dot_matrix = matrix @ matrix.T

    # 대각성분이 1, 그 외는 -1/(c-1)인 matrix
    rho = torch.full((c, c), -1/(c-1), device=device)
    rho.fill_diagonal_(1.0)
    
    S = matrix @ matrix.t()

    return ((S - rho) ** 2).mean()

def sim_function(h, mu, z, labels):
    # broadcast한 mu
    mu_br = mu[labels]

    sim_1 = cos_sim(h, z).mean() 
    sim_2 = cos_sim(mu_br, z).mean()

    return sim_1 + sim_2

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return F_loss.mean()
        else:
            return F_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='', help='Experiment run name')
    parser.add_argument('--mode', type=str, default='', help='Training mode')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--imb_ratio', type=int, default=1)
    args = parser.parse_args()

    save_dir = os.path.join('results', f"{args.run_name}_{args.imb_ratio}", args.mode)
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader, test_loader = load_cifar10(args)

    # Hyper Parameter
    Epochs = 100
    LR = 0.01
    device = 'cuda'

    C = args.num_classes

    print(f"****run_name    : {args.run_name}")
    print(f"****mode        : {args.mode}")
    print(f"****imb_ratio   : {args.imb_ratio}")

    # ResNet50
    model = ResNet(Bottleneck, [3,4,6,3])
    model.to(device)
    mlp_model = MLP()
    mlp_model.to(device)

    optimizer = optim.SGD(list(model.parameters()) + list(mlp_model.parameters()), lr=LR, momentum=0.9, weight_decay=5e-3)
    ce_loss = nn.CrossEntropyLoss()
    focal_loss = FocalLoss()
    cos_sim = nn.CosineSimilarity(dim=1)

    for epoch in range(1, Epochs+1):
        model.train()
        mlp_model.train()

        total = 0
        correct = 0

        gamma = 1
        eta = 1 - (epoch / Epochs) ** gamma

        for (x_1, x_2), labels in tqdm(train_loader):
            # save_image(x_1[0], '/home/suno3534/data/AllINC/images/1.png')
            # save_image(x_2[0], '/home/suno3534/data/AllINC/images/2.png')

            x_1, x_2, labels = x_1.to(device), x_2.to(device), labels.to(device)

            if args.mode == 'basic':
                logits_1, _, _ = model(x_1)
                loss = ce_loss(logits_1, labels)

                outputs = logits_1

            else:
                # z는 feature, w는 가중치 벡터를 의미
                logits_1, z_1, w_1 = model(x_1)
                logits_2, z_2, w_2 = model(x_2)

                # print('z', z_1)
                # print('w', w_1)

                h_1 = mlp_model(z_1)
                h_2 = mlp_model(z_2)

                # print('h', h_1)

                mu_1 = calcul_feature_mean(z_1, labels, C)
                mu_2 = calcul_feature_mean(z_2, labels, C)

                mu_tilde = center_and_normalize((mu_1 + mu_2)/2, torch.cat([z_1, z_2], dim=0))

                loss_con = - (sim_function(h_1, mu_2, z_2, labels) + sim_function(h_2, mu_1, z_1, labels)) / 2

                loss_mu = P2P(mu_tilde, C)
                loss_w = (P2P(w_1, C) + P2P(w_2, C)) / 2

                loss_cls_1 = eta * F.cross_entropy(logits_1, labels) + (1-eta) * (focal_loss(logits_1, labels) + loss_w)
                loss_cls_2 = eta * F.cross_entropy(logits_2, labels) + (1-eta) * (focal_loss(logits_2, labels) + loss_w)

                alpha = 3

                loss = loss_cls_1 + loss_cls_2 + alpha * (loss_con + loss_mu)

                outputs = (logits_1 + logits_2) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

        model.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, features, weight_vec = model(inputs)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            test_acc = 100. * correct / total

        if epoch % 1 == 0:
            print(f'Epoch: [{epoch}/{Epochs}] | Train Accuracy: {train_acc:.2f}% | Test Accuracy: {test_acc:.2f}% | Train Loss: {loss:.5f}')
            if args.mode != 'basic':
                print(f'{loss_cls_1.item():.3f} | {loss_cls_2.item():.3f} | {loss_w.item():.3f} | {loss_con.item():.3f} | {loss_mu.item():.3f}')

        if epoch % 20 == 0:
            save_path = os.path.join(save_dir, f'model_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
