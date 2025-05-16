import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
from collections import defaultdict

# Constants for label normalization
LABEL_MEAN = 500.0
LABEL_STD = 200.0

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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(repeat_channels)
])

aug_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=25),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
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
        norm_label = (label - LABEL_MEAN) / LABEL_STD
        return image, torch.tensor(norm_label, dtype=torch.float)

class AugmentedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        norm_label = (self.labels[idx] - LABEL_MEAN) / LABEL_STD
        return self.images[idx], torch.tensor(norm_label, dtype=torch.float)

def stratified_oversample(dataset, num_bins=10):
    bins = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        bin_id = int(min(label // (1000 / num_bins), num_bins - 1))
        bins[bin_id].append(idx)

    max_bin_size = max(len(idxs) for idxs in bins.values())
    balanced_indices = []

    for idxs in bins.values():
        if len(idxs) < max_bin_size:
            idxs = np.random.choice(idxs, size=max_bin_size, replace=True)
        balanced_indices.extend(idxs)

    return Subset(dataset, balanced_indices)

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

    # Balance original dataset
    balanced_train_dataset = stratified_oversample(train_dataset)

    # Augmentations
    augmented_images, augmented_labels = augment_dataset(raw_train, num_augmentations=20)
    augmented_dataset = AugmentedDataset(augmented_images, augmented_labels)

    combined_dataset = ConcatDataset([balanced_train_dataset, augmented_dataset])

    print(f"Original training data size: {len(train_dataset)}")
    print(f"Augmented samples added: {len(augmented_dataset)}")
    print(f"Total training data size: {len(combined_dataset)}")

    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def train_model(train_loader, test_loader, num_epochs=1, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads = nn.Sequential(
        nn.LayerNorm(model.heads.head.in_features),
        nn.Linear(model.heads.head.in_features, 1)
    )

    model = model.to(device)

    # Unfreeze top layers and head
    for name, param in model.named_parameters():
        if 'encoder.layers.10' in name or 'encoder.layers.11' in name or 'heads' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    criterion = nn.SmoothL1Loss()
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
        scheduler.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = (np.array(all_preds) * LABEL_STD) + LABEL_MEAN
    all_labels = (np.array(all_labels) * LABEL_STD) + LABEL_MEAN

    mae = mean_absolute_error(all_labels, all_preds)
    rmse = root_mean_squared_error(all_labels, all_preds)
    print(f"\nMAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    return model, all_preds, all_labels

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='b')
    plt.plot(np.unique(y_true), np.poly1d(np.polyfit(y_true, y_pred, 1))(np.unique(y_true)), color='red')
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    r_squared = np.corrcoef(y_true, y_pred)[0, 1] ** 2
    plt.text(0.05, 0.95, f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_squared:.2f}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
    plt.xlabel("True Cycle Count (0 to 1000)")
    plt.ylabel("Predicted Cycle Count")
    plt.title("ViT Regression: True vs Predicted Cycles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==== Visualize attention on one test image ====
import torchvision.transforms.functional as TF
import cv2

def visualize_attention_map(model, image_tensor):
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.unsqueeze(0).to(device)

    attention_maps = []

    def hook_fn(module, input, output):
        attention_maps.append(module.attn_output_weights.detach())

    handle = model.encoder.layers[-1].self_attn.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(image_tensor)

    handle.remove()

    if not attention_maps:
        print("No attention maps were collected.")
        return

    attn = attention_maps[0][0].mean(dim=0)[0, 1:]  # shape: (seq_len - 1,)
    size = int(attn.shape[0] ** 0.5)
    attn = attn[:size * size].reshape(size, size).cpu().numpy()

    # Resize attention to image size
    attn_resized = cv2.resize(attn, (224, 224), interpolation=cv2.INTER_CUBIC)
    attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_norm), cv2.COLORMAP_JET)

    # Prepare input image for overlay (convert tensor to numpy image)
    img_np = image_tensor.squeeze(0).cpu()
    if img_np.shape[0] == 3:
        img_np = img_np.permute(1, 2, 0)
    else:
        img_np = img_np.squeeze(0)
        img_np = np.stack([img_np] * 3, axis=-1)
    img_np = ((img_np * 0.5 + 0.5) * 255).numpy().astype(np.uint8)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Plot side by side
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Attention Overlay")
    plt.axis('off')

    plt.suptitle("ViT Attention Visualization", fontsize=14)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    root_dir = 'D:\\jungha\\2025 Spring\\MEC510\\term project\\Processed_Data\\manmade'
    train_loader, test_loader = load_dataset(root_dir)
    model, y_pred, y_true = train_model(train_loader, test_loader, num_epochs=100)

    plot_predictions(y_true, y_pred)

    # ==== Save model ====
    torch.save(model.state_dict(), "vit_cycle_predictor.pt")
    print("Saved model to vit_cycle_predictor.pt")

    # example_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
    # traced_model = torch.jit.trace(model, example_input)
    # traced_model.save("vit_cycle_predictor_traced.pt")
    # print("Saved TorchScript model to vit_cycle_predictor_traced.pt")    

    # sample_img, _ = next(iter(test_loader))
    # visualize_attention_map(model, sample_img[0])

