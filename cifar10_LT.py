import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image


base_transform = transforms.Compose([
    transforms.ToTensor()
])
transform_1 = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform_2 = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def make_cifar10_lt(root, imb_ratio, is_train, download):
    num_classes = 10

    # train 데이터셋은 imbalance하게 생성
    if is_train:
        data_1 = CIFAR10(root=root, train=True, download=download, transform=transform_1)
        data_2 = CIFAR10(root=root, train=True, download=download, transform=transform_2)
        
        max_sample = 5000
        min_sample = int(max_sample / imb_ratio)

        # 지수 감소 계수: mu^(C-1) = min_sample / max_sample
        mu = (min_sample / max_sample) ** (1.0 / (num_classes - 1))
        
        img_num_per_cls = [int(max_sample * (mu ** cls_idx)) for cls_idx in range(num_classes)]

        targets_np = np.array(data_1.targets)

        selected_indices = []
        for cls_idx, n_imgs in enumerate(img_num_per_cls):
            cls_indices = np.where(targets_np == cls_idx)[0]
            np.random.shuffle(cls_indices)
            selected_indices.extend(cls_indices[:n_imgs].tolist())

        dataset_1 = Subset(data_1, selected_indices)
        dataset_2 = Subset(data_2, selected_indices)

        data_list = [((dataset_1[i][0], dataset_2[i][0]), dataset_1[i][1]) for i in range(len(dataset_1))]

        dataset = ListDataset(data_list)

    # test 데이터셋은 그대로 생성
    else:
        dataset = CIFAR10(root=root, train=False, download=download, transform=transform_1)
    
    return dataset
