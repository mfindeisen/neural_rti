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
    output_dir = "output"
    weights_path = os.path.join(output_dir, "decoder_weights.json")
    latent_path = os.path.join(output_dir, "latent_map.png")
    
    docs_public = "C:/Users/m/Projects/modernRtiViewer/docs/public"
    os.makedirs(docs_public, exist_ok=True)
    
    print("Loading latent map...")
    img = Image.open(latent_path)
    w, h = img.size
    
    # Get raw NRGBA pixels using numpy to bypass PIL's transparency/alpha handling
    latent_raw = np.array(img, dtype=np.uint8)
    
    # Downsample to 800px width (maintain aspect ratio)
    target_width = 800
    target_height = int(h * target_width / w)
    
    # Pure mathematical nearest neighbor downsampling
    scale_factor = w / target_width
    indices_y = (np.arange(target_height) * scale_factor).astype(np.int32)
    indices_x = (np.arange(target_width) * scale_factor).astype(np.int32)
    
    # Select grid
    latent_resized_np = latent_raw[indices_y][:, indices_x]
    
    # Save latent map preview (forcing alpha to 255 so browser does not alpha-blend it with the page background)
    preview_rgba = latent_resized_np.copy()
    preview_rgba[:, :, 3] = 255
    img_preview = Image.fromarray(preview_rgba, mode="RGBA")
    img_preview.save(os.path.join(docs_public, "latent_map_preview.png"))
    print("Saved latent map preview.")
    
    print("Evaluating MLP for reconstruction preview...")
    # Load weights
    decoder = load_decoder_from_json(weights_path)
    
    # Run reconstruction using raw, un-premultiplied latent map values
    latent_float = latent_resized_np.astype(np.float32) / 255.0
    H, W, D = latent_float.shape
    latent_flat = torch.from_numpy(latent_float.reshape(-1, D))
    
    # Light direction straight from above [0, 0, 1]
    light_dir = torch.tensor([0.0, 0.0, 1.0]).expand(H * W, 3)
    
    with torch.no_grad():
        pred_rgb = decoder(latent_flat, light_dir)
        pred_rgb_np = (pred_rgb.numpy().reshape(H, W, 3) * 255.0).astype(np.uint8)
        
    img_recon = Image.fromarray(pred_rgb_np)
    img_recon.save(os.path.join(docs_public, "reconstruction_preview.jpg"), quality=90)
    print("Saved reconstruction preview.")

if __name__ == "__main__":
    main()
