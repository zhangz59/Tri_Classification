from medmnist import OrganMNIST3D
# Load datasets

dataset_val = OrganMNIST3D(split="val", download=True,root="./medmnist")
dataset_train = OrganMNIST3D(split="train", download=True,root="./medmnist")
dataset_test = OrganMNIST3D(split="test", download=True,root="./medmnist")




print(dataset_train.size)