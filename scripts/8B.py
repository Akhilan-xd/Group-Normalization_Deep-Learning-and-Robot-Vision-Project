import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# Dataset directory
dataset_dir = '/home/aashok2s/DLRV/GN_vs_BN/tiny-imagenet-200'
train_dir = os.path.join(dataset_dir, 'train')
# for root, dirs, files in os.walk(train_dir):
#     print(f"Found directory: {root}")
#     for file in files:
#         print(f"\tFile: {file}")
val_dir = os.path.join(dataset_dir, 'val')

# Data augmentation and normalization for training
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageFolder(train_dir, transform=transform_train)
val_dataset = ImageFolder(val_dir, transform=transform_val)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=200, norm_layer=None, num_groups=None):
        super(SimpleCNN, self).__init__()
        print(f'[Simple_CNN] --> Initialization')
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            self._get_norm_layer(norm_layer, 32, [32, 64, 64] if norm_layer == nn.LayerNorm else None),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            self._get_norm_layer(norm_layer, 64, [64, 32, 32] if norm_layer == nn.LayerNorm else None),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            self._get_norm_layer(norm_layer, 128, [128, 16, 16] if norm_layer == nn.LayerNorm else None),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Increase dropout for better regularization
            nn.Linear(256, num_classes)
        )

    def _get_norm_layer(self, norm_layer, num_features, normalized_shape=None):
        if norm_layer == nn.GroupNorm:
            return norm_layer(num_groups, num_features)
        elif norm_layer == nn.LayerNorm:
            return norm_layer(normalized_shape)
        elif norm_layer == nn.InstanceNorm2d:
            return norm_layer(num_features, affine=True)
        elif norm_layer == nn.BatchNorm2d:
            return norm_layer(num_features)
        else:
            return nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = self.classifier(x)
        return x

import time

def train_model(model_class, norm_layer=None, num_groups=None, batch_size=128, num_epochs=10):
    model = model_class(norm_layer=norm_layer, num_groups=num_groups).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    print(f'[Training_Model] --> INITIALIZED')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    epoch_losses = []
    model.train()
    
    # Track the start time
    start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # Calculate total time taken
    total_time = time.time() - start_time
    print(f'Total time for training: {total_time:.2f} seconds')
    
    return epoch_losses, total_time

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters for the training
batch_size = 8
num_epochs = 10

# Compare training times for SimpleCNN
normalizations = [
    ('BatchNorm', nn.BatchNorm2d, None),
    ('LayerNorm', nn.LayerNorm, None),
    ('InstanceNorm', nn.InstanceNorm2d, None),
    ('GroupNorm_32', nn.GroupNorm, 32),
    ('GroupNorm_16', nn.GroupNorm, 16),
    ('GroupNorm_8', nn.GroupNorm, 8),
    ('GroupNorm_4', nn.GroupNorm, 4),
    ('GroupNorm_2', nn.GroupNorm, 2)
]

results_complex = {norm_name: [] for norm_name, _, _ in normalizations}

for norm_name, norm_layer, num_groups in normalizations:
    print(f'Normalization: {norm_name}')
    epoch_losses = train_model(SimpleCNN, norm_layer=norm_layer, num_groups=num_groups, batch_size=batch_size, num_epochs=num_epochs)
    results_complex[norm_name] = epoch_losses

# To save the figure
folder_path_1 = '/home/aashok2s/DLRV/GN_vs_BN/Images'
file_name_1 = '8_B_Training loss vs epochs.png'
file_path_1 = os.path.join(folder_path_1, file_name_1)

folder_path_2 = '/home/aashok2s/DLRV/GN_vs_BN/Images'
file_name_2 = '8_B_Training time vs Normalization type.png'
file_path_2 = os.path.join(folder_path_2, file_name_2)


# List to store training times for each normalization method
train_times = []

# Train and compare normalization methods with SimpleCNN
for norm_name, norm_layer, num_groups in normalizations:
    print(f'Normalization: {norm_name}')
    epoch_losses, total_time = train_model(SimpleCNN, norm_layer=norm_layer, num_groups=num_groups, batch_size=batch_size, num_epochs=num_epochs)
    
    # Store the results and the training time
    results_complex[norm_name] = epoch_losses
    train_times.append(total_time)
plt.figure(figsize=(10,6))
for norm_name, losses in results_complex.items():
    plt.plot(range(num_epochs), losses, label=norm_name)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epoch for Different Normalizations (SimpleCNN)')
plt.legend()
plt.savefig(file_path_1)
plt.show()

# Plot the time taken vs. normalization type
plt.figure(figsize=(10, 6))
plt.bar([norm_name for norm_name, _, _ in normalizations], train_times, color='skyblue')
plt.xlabel('Normalization Type')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time vs. Normalization Type (SimpleCNN)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(file_path_2)
# Show the plot
plt.show()






