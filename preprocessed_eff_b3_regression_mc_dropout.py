
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import EfficientNet_B3_Weights
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

base_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

aug_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=25),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
])

class TensorLabelWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, torch.tensor(label, dtype=torch.float32)

class AugmentedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

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
    raw_train = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform=base_transform)

    train_dataset = TensorLabelWrapper(datasets.ImageFolder(os.path.join(root_dir, 'train'), transform=base_transform))
    test_dataset = datasets.ImageFolder(os.path.join(root_dir, 'test'), transform=base_transform)

    augmented_images, augmented_labels = augment_dataset(raw_train, num_augmentations=5)
    augmented_dataset = AugmentedDataset(augmented_images, augmented_labels)

    combined_dataset = ConcatDataset([train_dataset, augmented_dataset])

    print(f"Original training data size: {len(train_dataset)}")
    print(f"Augmented samples added: {len(augmented_dataset)}")
    print(f"Total training data size: {len(combined_dataset)}")

    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader, raw_train.classes

def enable_mc_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def train_model(train_loader, test_loader, num_epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(model.classifier[1].in_features, 1)
    )
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")
        scheduler.step()

    return model

def evaluate_with_uncertainty(model, test_loader, num_samples=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    enable_mc_dropout(model)

    all_means = []
    all_stds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            predictions = []

            for _ in range(num_samples):
                outputs = model(images).squeeze().cpu().numpy()
                predictions.append(outputs)

            predictions = np.stack(predictions)
            pred_mean = predictions.mean(axis=0)
            pred_std = predictions.std(axis=0)

            all_means.extend(pred_mean)
            all_stds.extend(pred_std)
            all_labels.extend(labels.numpy())

    return np.array(all_means), np.array(all_stds), np.array(all_labels)

def show_regression_results(y_true, y_pred, y_std, class_names):
    y_pred_rounded = np.clip(np.round(y_pred), 0, 10).astype(int)
    y_true_int = np.array(y_true).astype(int)
    cm = confusion_matrix(y_true_int, y_pred_rounded)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title("Regression Confusion Matrix (MC Dropout)")
    plt.tight_layout()
    plt.show()

    # Plot predicted mean ± std
    plt.errorbar(range(len(y_pred)), y_pred, yerr=y_std, fmt='o', alpha=0.5)
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted Class (0–10)")
    plt.title("MC Dropout Predictions with Uncertainty")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    root_dir = 'D:\\jungha\\2025_Spring\\MEC510\\term_project\\Processed_Data\\manmade'
    train_loader, test_loader, class_names = load_dataset(root_dir)
    model = train_model(train_loader, test_loader)
    y_pred, y_std, y_true = evaluate_with_uncertainty(model, test_loader)
    show_regression_results(y_true, y_pred, y_std, class_names)
