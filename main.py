import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
import torch_geometric

#importing the GraphSAGE model and utility functions
from gnn_model import GNNModel
from data_processing import load_file, preprocess_graph

#hyperparameters
input_dim = 4       #number of input features per node (ex. vote_average, vote_count, revenue, release_date etc)
hidden_dim = 64     #size of hidden layers 
output_dim = 10     # number of output features
num_layers = 2      #number of layers in GraphSAGE model
learning_rate = 0.001   #learning rate for the optimizer 
batch_size = 64     #batch size for training
num_epochs = 20     #number of epochs to train

# loading dataset and preprocessing the graph

# Define the file path correctly
file_path = '/Users/shreyasravi/Desktop/gnn_recommender_system/data_file/movie_data_df.csv'

# Call the function with the file path
df, node_features = load_file(file_path)
edge_index = preprocess_graph('/Users/shreyasravi/Desktop/gnn_recommender_system/data_file/movie_data_df.csv')

#create a dataset for pytorch geometric
data = torch_geometric.data.Data(x=node_features, edge_index=edge_index)
data = [data]

# dataloader for batching
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# intialize GraphSAGE model
model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim= output_dim, num_layers=num_layers)

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss() # crossentropy for classification


# training loop

def train():
    model.train()
    total_loss = 0

    for batch in data_loader:
        optimizer.zero_grad()

        # Ensure that the edge_index is correctly provided
        out = model(batch.x, batch.edge_index)

        # Assuming target is included in the batch data
        target = batch.y  # Adjust according to your batch structure

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Average loss: {avg_loss:.4f}")


# Main training loop
if __name__ == "__main__":
    # Load the graph data
    data = preprocess_graph(file_path)

    # Initialize the model, criterion, and optimizer
    input_dim = data.x.shape[1]  # Number of input features
    output_dim = output_dim  # Set this according to your classification problem

    model = GNNModel(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()  # Use appropriate loss function
    num_epochs = 20  # Set number of epochs

    # Print edge index shape for debugging
    print(f"Edge index shape: {data.edge_index.shape}")

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train()

