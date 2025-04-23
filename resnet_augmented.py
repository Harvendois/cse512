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
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights, ResNet34_Weights, EfficientNet_B7_Weights
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
 
root_dir = 'D:\\jungha\\2025 Spring\\MEC510\\term project\\Processed_Data\\manmade'

################################ IMAGE PREPROCESSING ################################ 

# Transformation for all images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # ← LAMBDA is not pickleable
])

##################### LOAD DATASET #####################

def repeat_channels(x):
    return x.repeat(3, 1, 1)

# Load full dataset using ImageFolder
def load_full_dataset(root_dir, batch_size=32, num_workers=0):
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset.classes

######################################################## DATA AUGMENTATION ########################################################
# augmenting the dataset using flipping, rotation using data loaded from function load_full_dataset

def augment_dataset(dataset, num_augmentations=5):
    augmented_images = []
    augmented_labels = []

    for img, label in dataset:
        for _ in range(num_augmentations):
            aug_img = img

            # Random horizontal flip
            if random.random() > 0.5:
                aug_img = transforms.functional.hflip(aug_img)

            # Random rotation
            angle = random.randint(-25, 25)
            aug_img = transforms.functional.rotate(aug_img, angle)

            #Random affine
            aug_img = transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))(aug_img)

            #Gaussian blur
            if random.random() > 0.7:
                aug_img = transforms.GaussianBlur(kernel_size=3)(aug_img)

            # mixup 
            # if random.random() > 0.5:
            #     alpha = 0.2
            #     beta = 0.2
            #     lam = np.random.beta(alpha, beta)
            #     rand_index = random.randint(0, len(dataset) - 1)
            #     img2, label2 = dataset[rand_index]
            #     aug_img = lam * img + (1 - lam) * img2
            #     label = lam * label + (1 - lam) * label2

            # # Add Gaussian noise
            # noise = torch.randn_like(aug_img) * 0.05
            # aug_img = aug_img + noise
            # aug_img = torch.clamp(aug_img, 0.0, 1.0)

            augmented_images.append(aug_img)
            augmented_labels.append(label)

    return augmented_images, augmented_labels


####################################### MODEL DEFINITION #######################################

# Train and evaluate model
def train_resnet_model(train_loader, test_loader, num_classes, num_epochs=40, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model = model.to(device) 

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-4)

    # Use CosineAnnealingLR for learning rate scheduling
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # patience=3 means "wait 3 epochs without improvement"
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

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

        # Update the learning rate
        # scheduler.step()
        
        train_acc = correct / total * 100
        print(f"Train Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Evaluation
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

        scheduler.step(test_acc)

    return model, all_preds, all_labels


# Plot confusion matrix
def show_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

class TensorLabelWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, torch.tensor(label, dtype=torch.long)  # Ensure label is a tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Check if CUDA is available and print GPU information
print("CUDA Available:", torch.cuda.is_available())
print("GPU Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")


######################### MAIN ##########################

train_loader, test_loader, class_names = load_full_dataset(root_dir)


# Augment the dataset
augmented_images, augmented_labels = augment_dataset(train_loader.dataset, num_augmentations=5)
augmented_dataset = [(img, label.item() if isinstance(label, torch.Tensor) else label) for img, label in zip(augmented_images, augmented_labels)]
augmented_images_tensor = torch.stack(augmented_images)
augmented_labels_tensor = torch.tensor(augmented_labels, dtype=torch.long)
augmented_dataset = torch.utils.data.TensorDataset(augmented_images_tensor, augmented_labels_tensor)
wrapped_train_dataset = TensorLabelWrapper(train_loader.dataset)
combined_dataset = torch.utils.data.ConcatDataset([wrapped_train_dataset, augmented_dataset])

print(f"Original dataset size: {len(train_loader.dataset)}")
print(f"Augmented dataset size: {len(augmented_images)}")
print(f"Combined dataset size: {len(combined_dataset)}")


# Create a new DataLoader for the combined dataset
train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=0)
# Train and evaluate the model
model, y_pred, y_true = train_resnet_model(train_loader, test_loader, num_classes=len(class_names))
show_confusion_matrix(y_true, y_pred, class_names)
