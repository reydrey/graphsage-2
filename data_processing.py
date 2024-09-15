import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

def load_file(file_path):
    print(f"Loading file from {file_path}")
    df = pd.read_csv(file_path)
    print(f"File loaded. Shape: {df.shape}")

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
    if 'genre' in df.columns:
        df['genre'] = df['genre'].astype('category').cat.codes  # Convert genres to numerical labels
        target = torch.tensor(df['genre'].values, dtype=torch.long)
    else:
        target = torch.zeros(len(df), dtype=torch.long)  # Dummy target

    print(f"Node features shape: {node_features.shape}")
    print(f"Target shape: {target.shape}")

    return df, node_features, target

def create_edges(df, threshold=1000):
    print("Creating edges...")
    features = df[['vote_average', 'vote_count', 'revenue', 'release_date']].values
    nn_model = NearestNeighbors(n_neighbors=min(10, len(features)), radius=threshold).fit(features)
    distances, indices = nn_model.kneighbors(features)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i < j:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    print(f"Edge index shape: {edge_index.shape}")

    return edge_index

def preprocess_graph(file_path, threshold=1000, sample_size=10000):
    print("Preprocessing graph...")
    
    df, node_features, target = load_file(file_path)

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        node_features = node_features[df.index]
        target = target[df.index]

    edge_index = create_edges(df, threshold)

    data = Data(x=node_features, edge_index=edge_index, y=target)
    
    print("Graph preprocessing completed.")
    print(f"Data object: {data}")

    num_classes = target.max().item() + 1  # Number of unique classes
    print(f"Number of classes: {num_classes}")

    return data, num_classes
