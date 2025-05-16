import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm

# Set random seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def repeat_channels(x):
    return x.repeat(3, 1, 1)

# Transforms
base_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # [0, 1] range
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(repeat_channels)
])

aug_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=25),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(repeat_channels)
])

class FloatLabelWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, torch.tensor(label / 10.0, dtype=torch.float)

class AugmentedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(self.labels[idx] / 10.0, dtype=torch.float)

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def augment_dataset(dataset, num_augmentations):
    augmented_images = []
    augmented_labels = []
    raw_transform = dataset.transform
    dataset.transform = None

    for img, label in dataset:
        for _ in range(num_augmentations):
            aug_img = aug_transform(img)
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    dataset.transform = raw_transform
    return augmented_images, augmented_labels

def load_dataset(root_dir, batch_size=32):
    raw_train = datasets.ImageFolder(os.path.join(root_dir, 'train'))
    test_dataset = datasets.ImageFolder(os.path.join(root_dir, 'test'), transform=base_transform)

    train_dataset = FloatLabelWrapper(datasets.ImageFolder(os.path.join(root_dir, 'train'), transform=base_transform))

    augmented_images, augmented_labels = augment_dataset(raw_train, num_augmentations=10)
    augmented_dataset = AugmentedDataset(augmented_images, augmented_labels)

    combined_dataset = ConcatDataset([train_dataset, augmented_dataset])

    print(f"Original training data size: {len(train_dataset)}")
    print(f"Augmented samples added: {len(augmented_dataset)}")
    print(f"Total training data size: {len(combined_dataset)}")

    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader

def train_model(train_loader, test_loader, num_epochs=50, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads = nn.Sequential(
        nn.LayerNorm(model.heads.head.in_features),
        nn.Linear(model.heads.head.in_features, 1)
    )
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")
        scheduler.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds) * 1000
    all_labels = np.array(all_labels) * 100

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = root_mean_squared_error(all_labels, all_preds)
    print(f"\nMAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    return model, all_preds, all_labels

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='b')
    plt.plot(np.unique(y_true), np.poly1d(np.polyfit(y_true, y_pred, 1))(np.unique(y_true)), color='red')
    # extract the linear regression line slope and R^2 value
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    r_squared = np.corrcoef(y_true, y_pred)[0, 1] ** 2
    plt.text(0.05, 0.95, f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_squared:.2f}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
    # plt.xlim(0, 1000)
    # plt.ylim(0, 1000)
    plt.xlabel("True Cycle Count (0 to 1000)")
    plt.ylabel("Predicted Cycle Count")
    plt.title("ViT Regression: True vs Predicted Cycles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_attention(model, image_tensor):
    model.eval()
    attention_maps = []
    hooks = []

    def hook_fn(module, input, output):
        if hasattr(output, 'attn_output_weights'):
            attention_maps.append(output.attn_output_weights)

    for block in model.encoder.layers:
        hooks.append(block.attention.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0))

    for h in hooks:
        h.remove()

    if attention_maps:
        last_attention = attention_maps[-1][0]  # shape: (num_heads, seq_len, seq_len)
        mean_attention = last_attention.mean(dim=0)[0, 1:]  # avg over heads, ignore [CLS]
        num_patches = int((image_tensor.shape[1] * image_tensor.shape[2]) / (16 * 16))
        size = int(num_patches ** 0.5)
        attention_map = mean_attention[:num_patches].reshape(size, size).cpu().numpy()
        plt.imshow(attention_map, cmap='viridis')
        plt.colorbar()
        plt.title("ViT Attention Map (Last Layer)")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    root_dir = 'D:\\jungha\\2025 Spring\\MEC510\\term project\\Processed_Data\\manmade'
    train_loader, test_loader = load_dataset(root_dir)
    model, y_pred, y_true = train_model(train_loader, test_loader)
    plot_predictions(y_true, y_pred)

