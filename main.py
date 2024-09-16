import sys
import os

# Disable memory profiler
sys.modules['memory_profiler'] = None
os.environ['PYTHON_MEMORY_PROFILER_DISABLE'] = 'True'

import torch
import torch.nn as nn
from torch_geometric.data import Data

# Importing the GraphSAGE model and utility functions
from gnn_model import GNNModel
from data_processing import preprocess_graph

# Hyperparameters
hidden_dim = 64         # Size of hidden layers 
num_layers = 3          # Number of layers in GraphSAGE model
learning_rate = 0.001   # Learning rate for the optimizer 
num_epochs = 20         # Number of epochs to train
output_dim = 10         # Number of output classes or dimensions (e.g., number of recommendations)


# Define the file path correctly
file_path = '/Users/shreyasravi/Desktop/gnn_recommender_system/gnn_recommender/data_file/movie_data_df.csv'

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    
    # Ensure that output and target shapes match
    output = out[data.train_mask]
    target = data.y[data.train_mask]
    
    # Compute loss
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()
    return acc


def main():
    print("Starting main function...")
    
    # Load and preprocess the graph data
    print("Preprocessing graph data...")
    data, num_classes = preprocess_graph(file_path)
    print("Graph data preprocessed.")
    
    # Ensure data is a PyTorch Geometric Data object
    if not isinstance(data, Data):
        raise TypeError("preprocess_graph should return a PyTorch Geometric Data object")
    
    # Set input_dim and output_dim based on the data
    input_dim = data.num_node_features
    output_dim = num_classes  # Use the number of unique classes

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")

    # Initialize the model, criterion, and optimizer
    print("Initializing model...")
    model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Use BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    print("Model initialized.")

    # Print data information for debugging
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of node features: {data.num_node_features}")
    print(f"Number of classes: {output_dim}")

    # Training loop
    print("Starting training loop...")
    for epoch in range(num_epochs):
        loss = train(model, data, optimizer, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

        # Evaluate the model every few epochs
        if (epoch + 1) % 5 == 0:
            acc = evaluate(model, data)
            print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {acc:.4f}")

    print("Training completed.")


if __name__ == "__main__":
    main()
