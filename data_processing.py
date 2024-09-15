import torch
import pandas as pd
from torch_geometric.data import Data
from memory_profiler import profile

import torch
import pandas as pd

def load_file(file_path):
    """
    Load the dataset from the CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    df (DataFrame): The DataFrame containing the data.
    node_features (Tensor): A tensor of node features extracted from the DataFrame.
    target (Tensor): A tensor of target labels extracted from the DataFrame.
    """
    # Load the dataset from the CSV file
    df = pd.read_csv(file_path)

    # Convert 'release_date' to a timestamp in seconds since epoch
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_date'] = df['release_date'].astype(int) / 1e9  # Convert to float in seconds

    # Extract node features; ensure they are numeric
    feature_columns = ['vote_average', 'vote_count', 'revenue', 'release_date']
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    node_features = df[feature_columns].values

    # Convert the extracted features into PyTorch tensors with dtype float
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Extract and process target labels
    # Example for classification (e.g., genre labels):
    if 'genre' in df.columns:
        df['genre'] = df['genre'].astype('category').cat.codes  # Convert genres to numerical labels
        target = torch.tensor(df['genre'].values, dtype=torch.long)
    else:
        # Default: No target column, or you can add a different target processing
        target = torch.zeros(len(df), dtype=torch.long)  # Dummy target

    return df, node_features, target

import torch
from sklearn.neighbors import NearestNeighbors
import logging

def create_edges(df, threshold=1000):
    vote_counts = df[['vote_count']].values
    nn_model = NearestNeighbors(n_neighbors=10, radius=threshold).fit(vote_counts)
    distances, indices = nn_model.kneighbors(vote_counts)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i < j:
                edge_index.append([i, j])

    # Convert to tensor and ensure shape is [2, num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Ensure that edge_index has the correct shape
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index should have shape [2, num_edges], but got {edge_index.shape}")

    return edge_index


@profile

def preprocess_graph(file_path, threshold=1000, sample_size=10000):
    """Pre-process the graph data by loading features, creating edges, and returning graph data."""

    # Load data and extract node features
    df, node_features = load_file(file_path)

    # Sample the dataset if it's larger than the sample size
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Create edges based on similarity criterion
    edge_index = create_edges(df, threshold)

    # Create PyTorch Geometric Data object
    data = Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=edge_index)

    return data



