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
from torchvision.models import EfficientNet_B3_Weights
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
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(repeat_channels)
])

aug_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=25),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
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
        return image, torch.tensor(label * 100.0, dtype=torch.float)  # Convert class index to cycle count

class AugmentedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(self.labels[idx] * 100.0, dtype=torch.float)

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

    augmented_images, augmented_labels = augment_dataset(raw_train, num_augmentations=5) # Number of augmentations per image        
    augmented_dataset = AugmentedDataset(augmented_images, augmented_labels)

    combined_dataset = ConcatDataset([train_dataset, augmented_dataset])

    print(f"Original training data size: {len(train_dataset)}")
    print(f"Augmented samples added: {len(augmented_dataset)}")
    print(f"Total training data size: {len(combined_dataset)}")

    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader

def train_model(train_loader, test_loader, num_epochs=30, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier[1].in_features, 1)
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
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = root_mean_squared_error(all_labels, all_preds)
    print(f"\nMAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    return model, all_preds, all_labels

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(np.array(y_true) * 100, y_pred, alpha=0.6, edgecolors='b')
    plt.plot([0, 1000], [0, 1000], 'r--')  # Ideal line
    plt.xlabel("True Cycle Count (0 to 1000)")
    plt.ylabel("Predicted Cycle Count")
    plt.title("Regression: True vs Predicted Cycles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    root_dir = 'D:\\jungha\\2025 Spring\\MEC510\\term project\\Processed_Data\\manmade'
    train_loader, test_loader = load_dataset(root_dir)
    model, y_pred, y_true = train_model(train_loader, test_loader)
    plot_predictions(y_true, y_pred)