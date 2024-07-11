from torch import nn

class BaseModel(nn.Module):
    def __init__(self, feature_dim, device) -> None:
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
    
    def preprocess(self, image):
        raise NotImplementedError
    
    def forward(self, img_tensor):
        raise NotImplementedError
     