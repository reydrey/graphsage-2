import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        """
        Initializing the GNN model
        
        input_dim = number of input features per node
        hidden_dim = number of hidden features in GNN layers
        output_dim = number of output features (number of classes or regression output)
        num_layers = number of layers in the GNN (default is 2)
        """
        super(GNNModel, self).__init__()

        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(pyg_nn.SAGEConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(pyg_nn.SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.convs.append(pyg_nn.SAGEConv(hidden_dim, output_dim))

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the network

        params:
            - x (tensor): node feature matrix of shape [num_nodes, num_features]
            - edge_index (tensor): edge indices of shape [2, num_edges]
            - batch (tensor, optional): batch tensor for pooling

        returns:
            - tensor: output features of the graph
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        if batch is not None:
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.fc(x)
        return x

"""
if __name__ == "__main__":
    # Define model parameters
    input_dim = 4      # Number of node features (e.g., vote average, vote count, revenue, etc.)
    hidden_dim = 64    # Number of hidden units in GraphSAGE layers
    output_dim = 10    # Number of output classes or dimensions (e.g., number of recommendations)
    num_layers = 3     # Number of layers in the GraphSAGE model (can be adjusted)

    # Create the GraphSAGE model instance
    model = GNNModel(input_dim, hidden_dim, output_dim, num_layers)

    # Print the model architecture
    print(model)
"""

        