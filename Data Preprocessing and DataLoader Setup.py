import torch
from torch.utils.data import DataLoader
from medmnist import OrganMNIST3D
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os

from model.MultiViewViT import MultiViewViT

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load datasets
dataset_train = OrganMNIST3D(split="train", root="./medmnist")
dataset_val = OrganMNIST3D(split="val", root="./medmnist")
dataset_test = OrganMNIST3D(split="test", root="./medmnist")

# Create DataLoaders
batch_size = 2

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Check the shape of data
images, labels = next(iter(train_loader))
print(f"Images batch shape: {images.shape}")
print(f"Labels batch shape: {labels.shape}")

# Example usage inside MultiViewViT
vit_args = {
    'emb_dim': 768,
    'mlp_dim': 3072,
    'num_heads': 12,
    'num_layers': 10,
    'dropout_rate': 0.1,
    'num_classes': 11,  # Number of output classes
}

# Instantiate the MultiViewViT model and move it to the device (GPU/CPU)
model = MultiViewViT(
    image_sizes=[(28, 28), (28, 28), (28, 28)],  # Size of each view (2D slices)
    patch_sizes=[(7, 7), (7, 7), (7, 7)],        # Patch sizes for ViT
    num_channals=[1, 1, 1],                      # Grayscale images (1 channel)
    vit_args=vit_args,
    mlp_dims=[33,64, 128,256,128,64, 11]             # MLP dimensions
).to(device)  # Move model to GPU/CPU

# Function to save the model
def save_model(model, epoch, val_accuracy, model_path="best_model.pth"):
    print(f"Saving model at epoch {epoch} with validation accuracy: {val_accuracy:.2f}%")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_accuracy': val_accuracy,
    }, model_path)

# Training loop with validation and model saving
def train_model(model, train_loader, val_loader, epochs=10, model_path="./best_model"):
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_accuracy = 0.0  # Track the best validation accuracy

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            labels = labels.squeeze(1)
            labels = labels.long()

            # Move data to GPU/CPU
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            loss = criterion(outputs, labels)


            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

        # Validate the model after each epoch
        val_accuracy = validate_model(model, val_loader)

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, epoch+1, best_val_accuracy, model_path)

def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU/CPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

# Run training
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, epochs=10, model_path="best_model.pth")

