import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

def load_file(file_path):
    print("Entered load_file function")
    try:
        df = pd.read_csv(file_path)
        print(f"File loaded. Shape: {df.shape}")

        # Print column names to debug
        print("Columns in the dataframe:", df.columns)

        # Convert 'release_date' to a timestamp in seconds since epoch
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_date'] = df['release_date'].astype('int64', errors='ignore') / 1e9  # Convert to float in seconds

        # Extract node features; ensure they are numeric
        feature_columns = ['vote_average', 'vote_count', 'revenue', 'release_date']
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        node_features = df[feature_columns].values
        node_features = torch.tensor(node_features, dtype=torch.float)

        # Handle genres for multi-label classification
        if 'genres' in df.columns:
            print("Handling missing values in 'genres' column...")
            df['genres'] = df['genres'].fillna('')  # Fill NaNs with empty string
            genres_list = df['genres'].apply(lambda x: x.split(', '))  # Split genres into lists
            mlb = MultiLabelBinarizer()
            target = mlb.fit_transform(genres_list)  # One-hot encode genres
            target = torch.tensor(target, dtype=torch.float)  # Use float for multi-label classification
        else:
            print("'genres' column not found in the dataframe.")
            target = torch.zeros(len(df), dtype=torch.float)

        print("Unique genres after processing:", mlb.classes_)
        print("Genre distribution after processing:", target.sum(dim=0))  # Sum of each genre across all movies
        print(f"Node features shape: {node_features.shape}")
        print(f"Target shape: {target.shape}")
        return df, node_features, target
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None


def create_edges(df, threshold=1000):
    print("Creating edges...")
    
    # Extract features for edge creation
    features = df[['vote_average', 'vote_count', 'revenue', 'release_date']].values

    # Fit NearestNeighbors model
    nn_model = NearestNeighbors(n_neighbors=min(10, len(features)), radius=threshold, algorithm='auto').fit(features)

    # Find distances and indices of neighbors
    distances, indices = nn_model.kneighbors(features)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i < j:
                edge_index.append([i, j])

    # Convert edge_index to tensor format suitable for PyTorch Geometric
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    print(f"Edge index shape: {edge_index.shape}")

    return edge_index


def split_data(num_nodes, test_size=0.2, val_size=0.1):
    # Split the data into train and test sets
    train_indices, test_indices = train_test_split(range(num_nodes), test_size=test_size, random_state=42)
    
    # Split the training data further into train and validation sets
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size / (1 - test_size), random_state=42)

    return train_indices, val_indices, test_indices


def preprocess_graph(file_path):
    print("Preprocessing graph...")
    
    # Load the data
    df, node_features, target = load_file(file_path)
    
    if df is None:
        raise ValueError("Failed to load the file.")
    
    print("Data loaded. Proceeding with graph preprocessing...")
    
    # Create edge_index
    edge_index = create_edges(df)  # Replace with create_edge_index if needed

    # Reshape target to be 2D for BCEWithLogitsLoss
    target = target.view(-1, target.size(-1))  # Ensures shape (N, C)
    target = target.float()  # Ensure target is of type float

    # Create the Data object for PyTorch Geometric
    data = Data(x=node_features, y=target, edge_index=edge_index)

    # Split data into train, validation, and test sets
    num_nodes = len(df)
    train_indices, val_indices, test_indices = split_data(num_nodes)
    
    # Create masks for training, validation, and testing
    data.train_mask = torch.tensor(train_indices, dtype=torch.long)
    data.val_mask = torch.tensor(val_indices, dtype=torch.long)
    data.test_mask = torch.tensor(test_indices, dtype=torch.long)

    num_classes = target.size(1)  # Number of classes for multi-label classification

    return data, num_classes

