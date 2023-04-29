from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

# Load a dataset
dataset = TUDataset(root='~/datasets/TUDataset', name='ENZYMES')

# Split the dataset into train and test sets
train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_dataset = dataset[train_idx]
test_dataset = dataset[test_idx]

# Create DataLoader objects for train and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
