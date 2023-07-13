import torch
import torch.nn as nn

def get_classifier(input_dims, hidden_dims=768, output_classes=200, n_layers=2):
    
    modules = []
    for i in range(n_layers):
        modules.append(nn.Linear(input_dims, hidden_dims))
        modules.append(nn.ReLU())
    
    modules.append(nn.Linear(hidden_dims, output_classes))
    modules.append(nn.Softmax())
    clf_model = nn.Sequential(*modules)
    
    return clf_model