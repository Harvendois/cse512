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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchvision.models import EfficientNet_B3_Weights
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)



# Ordinal-aware loss (distance-penalized Cross Entropy)
class OrdinalLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        base_loss = self.ce(logits, targets)
        pred_labels = torch.argmax(logits, dim=1)
        distance = torch.abs(pred_labels - targets).float()
        return (1 + distance) * base_loss

base_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

aug_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=25),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
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
        return image, torch.tensor(label, dtype=torch.long)

class AugmentedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(self.labels[idx], dtype=torch.long)

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
    # Load the raw training dataset without normalization for mean and std calculation
    raw_train = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform=transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ]))

    # Calculate mean and std for the training dataset
    mean, std = calculate_mean_std(raw_train)
    print(f"Calculated Mean: {mean}, Calculated Std: {std}")

    # Define transforms with calculated mean and std
    base_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[mean], std=[std])  # Use calculated values
    ])

    aug_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=25),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        #transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])  # Use calculated values
    ])

    # Reload datasets with updated transforms
    train_dataset = TensorLabelWrapper(datasets.ImageFolder(os.path.join(root_dir, 'train'), transform=base_transform))
    test_dataset = datasets.ImageFolder(os.path.join(root_dir, 'test'), transform=base_transform)

    # Augment the training dataset
    augmented_images, augmented_labels = augment_dataset(raw_train, num_augmentations=5)
    augmented_dataset = AugmentedDataset(augmented_images, augmented_labels)

    # Combine original and augmented datasets
    full_train_dataset = ConcatDataset([train_dataset, augmented_dataset])
    train_subset, val_subset = split_dataset(full_train_dataset)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    combined_dataset = ConcatDataset([train_dataset, augmented_dataset])

    print(f"Original training data size: {len(train_dataset)}")
    print(f"Augmented samples added: {len(augmented_dataset)}")
    print(f"Total training data size: {len(combined_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader, raw_train.classes

def train_model(train_loader, val_loader, test_loader, num_classes, num_epochs=25, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    model = model.to(device)

    criterion = OrdinalLoss(num_classes=num_classes)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total * 100
        print(f"Validation Accuracy: {val_acc:.2f}%")

        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # Evaluate
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = correct / total * 100
        print(f"Test Accuracy: {test_acc:.2f}%\n")
        scheduler.step(test_acc)

    return model, all_preds, all_labels

def show_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def calculate_mean_std(dataset):
    """
    Calculate the mean and standard deviation of pixel intensities across the dataset.
    Args:
        dataset: A PyTorch dataset object.
    Returns:
        mean: Mean pixel intensity.
        std: Standard deviation of pixel intensity.
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        # Flatten the images to calculate mean and std across all pixels
        images = images.view(images.size(0), -1)
        mean += images.mean(1).sum()
        std += images.std(1).sum()
        total_images += images.size(0)

    mean /= total_images
    std /= total_images
    return mean.item(), std.item()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("CUDA Available:", torch.cuda.is_available())
    print("GPU Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

    root_dir = 'D:\\jungha\\2025_Spring\\MEC510\\term_project\\Processed_Data\\manmade_preprocessed'
    train_loader, val_loader, test_loader, class_names = load_dataset(root_dir)
    model, y_pred, y_true = train_model(train_loader, val_loader, test_loader, num_classes=len(class_names))
    show_confusion(y_true, y_pred, class_names)