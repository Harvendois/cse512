import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

root_dir = 'D:\\jungha\\2025 Spring\\MEC510\\term project\\Processed_Data\\manmade'

# Hough transform + Gaussian preprocessing

def apply_hough_extra_channel(img_tensor):
    img_np = img_tensor.squeeze().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img_np, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=20, maxLineGap=5)

    # Draw lines on a blank canvas
    line_img = np.zeros_like(img_np)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), 255, 1)

    # Convert both to tensors
    original = torch.from_numpy(img_np).float() / 255.0
    hough = torch.from_numpy(line_img).float() / 255.0

    # Stack original and hough (2 channels)
    combined = torch.stack([original, hough], dim=0)

    # Ensure it returns 3 channels: original, hough, and a duplicate of either
    if combined.size(0) == 2:
        extra = combined[0].unsqueeze(0)  # repeat original as third channel
        combined = torch.cat([combined, extra], dim=0)
    return combined


# Transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: apply_hough_extra_channel(x)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
def load_full_dataset(root_dir, batch_size=32, num_workers=0):
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset.classes

# Augment dataset
def augment_dataset(dataset, num_augmentations=5):
    augmented_images = []
    augmented_labels = []

    for img, label in dataset:
        for _ in range(num_augmentations):
            aug_img = img
            if random.random() > 0.5:
                aug_img = transforms.functional.hflip(aug_img)
            angle = random.randint(-25, 25)
            aug_img = transforms.functional.rotate(aug_img, angle)
            aug_img = transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))(aug_img)
            noise = torch.randn_like(aug_img) * 0.05
            aug_img = torch.clamp(aug_img + noise, 0.0, 1.0)
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    return augmented_images, augmented_labels

# Custom wrapper to make sure labels are tensors
class TensorLabelWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, torch.tensor(label, dtype=torch.long)

# Train model
def train_resnet_model(train_loader, test_loader, num_classes, num_epochs=20, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model = model.to(device)

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = correct / total * 100
        print(f"Train Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = correct / total * 100
        print(f"Test Accuracy: {test_acc:.2f}%\n")

    return model, all_preds, all_labels

def show_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# MAIN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, class_names = load_full_dataset(root_dir)
augmented_images, augmented_labels = augment_dataset(train_loader.dataset, num_augmentations=5)
augmented_images_tensor = torch.stack(augmented_images)
augmented_labels_tensor = torch.tensor(augmented_labels, dtype=torch.long)
augmented_dataset = torch.utils.data.TensorDataset(augmented_images_tensor, augmented_labels_tensor)

wrapped_train_dataset = TensorLabelWrapper(train_loader.dataset)
combined_dataset = torch.utils.data.ConcatDataset([wrapped_train_dataset, augmented_dataset])

print(f"Original dataset size: {len(train_loader.dataset)}")
print(f"Augmented dataset size: {len(augmented_images)}")
print(f"Combined dataset size: {len(combined_dataset)}")

train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=0)
model, y_pred, y_true = train_resnet_model(train_loader, test_loader, num_classes=len(class_names))
show_confusion_matrix(y_true, y_pred, class_names)
