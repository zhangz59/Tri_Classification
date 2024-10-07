import torch
from torch.utils.data import DataLoader
from medmnist import OrganMNIST3D
from model.MultiViewViT import MultiViewViT

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


batch_size = 10

# Load datasets

dataset_test = OrganMNIST3D(split="test", root="./medmnist")


test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


# Load the saved model
vit_args = {
    'emb_dim': 768,
    'mlp_dim': 3072,
    'num_heads': 12,
    'num_layers': 10,
    'dropout_rate': 0.1,
    'num_classes': 11,  # Number of output classes
}
model = MultiViewViT(
    image_sizes=[(28, 28), (28, 28), (28, 28)],  # Size of each view (2D slices)
    patch_sizes=[(7, 7), (7, 7), (7, 7)],        # Patch sizes for ViT
    num_channals=[1, 1, 1],                      # Grayscale images (1 channel)
    vit_args=vit_args,
    mlp_dims=[33,64, 128,256,128,64, 11]             # MLP dimensions
).to(device)  # Move model to GPU/CPU


checkpoint = torch.load("./best_model/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)  # Move model to GPU/CPU

# Define the test function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Flatten the labels if necessary
            labels = labels.squeeze()

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Count correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Run the test
test_accuracy = test_model(model, test_loader)

if __name__ == "__main__":
    test_accuracy = test_model(model, test_loader)
