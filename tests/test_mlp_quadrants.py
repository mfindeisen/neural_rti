import os
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

class DecoderMLP(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )
        
    def forward(self, latent, light_dir):
        x = torch.cat([latent, light_dir], dim=-1)
        return self.net(x)

def load_decoder_from_json(json_path, latent_dim=4):
    with open(json_path, 'r') as f:
        weights = json.load(f)
    
    decoder = DecoderMLP(latent_dim=latent_dim)
    state_dict = decoder.state_dict()
    state_dict['net.0.weight'] = torch.tensor(weights['w1'])
    state_dict['net.0.bias'] = torch.tensor(weights['b1'])
    state_dict['net.2.weight'] = torch.tensor(weights['w2'])
    state_dict['net.2.bias'] = torch.tensor(weights['b2'])
    state_dict['net.4.weight'] = torch.tensor(weights['w3'])
    state_dict['net.4.bias'] = torch.tensor(weights['b3'])
    decoder.load_state_dict(state_dict)
    decoder.eval()
    return decoder

def main():
    weights_path = "neural_rti/output/decoder_weights.json"
    latent_path = "neural_rti/output/latent_map.png"
    
    if not os.path.exists(weights_path):
        print(f"Error: weights not found at {weights_path}")
        return
    if not os.path.exists(latent_path):
        print(f"Error: latent map not found at {latent_path}")
        return
        
    decoder = load_decoder_from_json(weights_path)
    
    latent_img = Image.open(latent_path).convert("RGBA")
    latent_np = np.array(latent_img, dtype=np.float32) / 255.0
    
    # Sample 1000 random latent vectors
    np.random.seed(42)
    H, W, D = latent_np.shape
    flat_latent = latent_np.reshape(-1, D)
    sampled_indices = np.random.choice(len(flat_latent), 1000, replace=False)
    latent_tensor = torch.from_numpy(flat_latent[sampled_indices])
    
    # Test directions
    directions = {
        "Top-Right (+0.6, +0.6, 0.529)": [0.6, 0.6, 0.529],
        "Top-Left  (-0.6, +0.6, 0.529)": [-0.6, 0.6, 0.529],
        "Bottom-Left(-0.6, -0.6, 0.529)": [-0.6, -0.6, 0.529],
        "Bottom-Right(+0.6, -0.6, 0.529)": [0.6, -0.6, 0.529]
    }
    
    print("--- PyTorch Model Predictions (Avg RGB) ---")
    for name, d in directions.items():
        d_tensor = torch.tensor([d], dtype=torch.float32).expand(1000, 3)
        with torch.no_grad():
            pred = decoder(latent_tensor, d_tensor)
            avg_rgb = torch.mean(pred, dim=0).numpy()
            min_rgb = torch.min(pred, dim=0)[0].numpy()
            max_rgb = torch.max(pred, dim=0)[0].numpy()
            print(f"{name}:")
            print(f"  Avg RGB: {avg_rgb}")
            print(f"  Min RGB: {min_rgb}")
            print(f"  Max RGB: {max_rgb}")

if __name__ == "__main__":
    main()
