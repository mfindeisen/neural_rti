import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

from neural_rti.reader import load_hsh

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

def sample_hemisphere_lights(num_lights):
    lights = []
    i = 0
    while len(lights) < num_lights:
        phi = np.pi * (3.0 - np.sqrt(5.0)) * i
        z = 1.0 - (i / float(num_lights)) * 0.95 
        radius = np.sqrt(max(0.0, 1.0 - z*z))
        x = np.cos(phi) * radius
        y = np.sin(phi) * radius
        
        length = np.sqrt(x*x + y*y + z*z)
        lights.append([x/length, y/length, z/length])
        i += 1
    return torch.tensor(lights, dtype=torch.float32)

def evaluate_hsh_torch(coeffs, bias, scale, light_dirs):
    device = coeffs.device
    scaled_coeffs = coeffs.float() / 255.0 * bias.to(device) + scale.to(device)
    
    lx, ly, lz = light_dirs[:, 0], light_dirs[:, 1], light_dirs[:, 2]
    cosTheta = lz
    cosTheta2 = cosTheta * cosTheta
    
    phi = torch.atan2(ly, lx)
    phi = torch.where(phi < 0, phi + 2 * np.pi, phi)
    cosPhi, sinPhi = torch.cos(phi), torch.sin(phi)
    
    l0 = 1.0 / np.sqrt(2.0 * np.pi)
    l1 = np.sqrt(6.0 / np.pi) * (cosPhi * torch.sqrt(torch.clamp(cosTheta - cosTheta2, min=0.0)))
    l2 = np.sqrt(3.0 / (2.0 * np.pi)) * (-1.0 + 2.0 * cosTheta)
    l3 = np.sqrt(6.0 / np.pi) * (torch.sqrt(torch.clamp(cosTheta - cosTheta2, min=0.0)) * sinPhi)
    
    L = torch.stack([torch.ones_like(l1) * l0, l1, l2, l3], dim=-1).to(device)
    reconstructed = torch.einsum('...ck,mk->m...c', scaled_coeffs, L)
    return torch.clamp(reconstructed, 0.0, 1.0)

def main():
    parser = argparse.ArgumentParser(description="Neural RTI PyTorch Training Pipeline (Random Sampling)")
    parser.add_argument("--input", type=str, required=True, help="Path to input .rti file")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="Random sampling steps per epoch")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=4, help="Dimension of latent space")
    parser.add_argument("--resize", type=int, default=0, help="Resize image. 0 = keep original.")
    parser.add_argument("--num-lights", type=int, default=64, help="Number of sampled light directions")
    parser.add_argument("--batch-size", type=int, default=262144, help="Number of random pixels per step (e.g. 512x512)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load RTI file
    print("Loading HSH file...")
    data = load_hsh(args.input)
    coeffs_np, bias_np, scale_np = data['coeffs'], data['bias'], data['scale']
    
    coeffs = torch.from_numpy(coeffs_np.copy()).to(device)
    bias = torch.from_numpy(bias_np.copy()).to(device)
    scale = torch.from_numpy(scale_np.copy()).to(device)
    
    H, W, C, K = coeffs.shape
    print(f"Loaded dimensions: {W}x{H}, channels: {C}, coefficients: {K}")
    
    if args.resize and max(H, W) > args.resize:
        scale_factor = args.resize / max(H, W)
        new_H, new_W = int(H * scale_factor), int(W * scale_factor)
        print(f"Downsampling to {new_W}x{new_H}...")
        coeffs_flat = coeffs.permute(2, 3, 0, 1).reshape(C*K, H, W).unsqueeze(0).float()
        coeffs_resized = nn.functional.interpolate(coeffs_flat, size=(new_H, new_W), mode='bilinear', align_corners=False)
        coeffs = coeffs_resized.squeeze(0).reshape(C, K, new_H, new_W).permute(2, 3, 0, 1).byte()
        H, W = new_H, new_W
        
    train_lights = sample_hemisphere_lights(args.num_lights).to(device)
    
    # 2. Initialize Latent Grid and Decoder MLP
    init_latent = coeffs.float().mean(dim=2) / 255.0 
    init_latent = torch.clamp(init_latent, 0.001, 0.999)
    init_logit = torch.log(init_latent / (1.0 - init_latent))
    
    latent_grid = nn.Parameter(init_logit.clone()) # Shape: (H, W, 4)
    decoder = DecoderMLP(latent_dim=args.latent_dim).to(device)
    
    optimizer = optim.Adam([
        {'params': [latent_grid], 'lr': args.lr},
        {'params': decoder.parameters(), 'lr': 0.001}
    ])
    criterion = nn.MSELoss()
    
    # Flatten data for easy random sampling
    num_pixels = H * W
    coeffs_flat = coeffs.view(num_pixels, C, K)
    
    # 3. Training Loop (Random Sampling)
    print(f"Starting training: {args.epochs} epochs, {args.steps_per_epoch} steps/epoch, {args.batch_size} pixels/step...")
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step in pbar:
            optimizer.zero_grad()
            
            # A) Pick 1 random light direction
            l_idx = torch.randint(0, args.num_lights, (1,)).item()
            l_dir = train_lights[l_idx:l_idx+1] # (1, 3)
            
            # B) Pick N random pixel indices
            pixel_indices = torch.randint(0, num_pixels, (args.batch_size,), device=device)
            
            # C) Extract data for these random pixels
            batch_coeffs = coeffs_flat[pixel_indices]
            
            # latent_grid has shape (H, W, 4), view it as flat to grab the right pixels
            batch_latent = latent_grid.view(num_pixels, args.latent_dim)[pixel_indices]
            
            # D) Ground Truth on the fly for these pixels
            with torch.no_grad():
                target_rgb = evaluate_hsh_torch(batch_coeffs, bias, scale, l_dir).squeeze(0)
            
            # E) MLP Prediction
            l_dir_expand = l_dir.expand(args.batch_size, 3)
            pred_rgb = decoder(torch.sigmoid(batch_latent), l_dir_expand)
            
            # F) Loss & Backprop
            loss = criterion(pred_rgb, target_rgb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if step % 10 == 0:
                pbar.set_postfix({'Loss': f"{loss.item():.6f}"})
                
    print("Training finished.")
    
    # 4. Export
    weights_dict = {}
    for name, param in decoder.named_parameters():
        if 'weight' in name:
            layer_idx = int(name.split('.')[1]) // 2 + 1
            weights_dict[f"w{layer_idx}"] = param.detach().cpu().numpy().tolist()
        elif 'bias' in name:
            layer_idx = int(name.split('.')[1]) // 2 + 1
            weights_dict[f"b{layer_idx}"] = param.detach().cpu().numpy().tolist()
            
    json_path = os.path.join(args.output_dir, "decoder_weights.json")
    with open(json_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    print("Saved decoder weights to:", json_path)
    
    # The latent grid retains its (H, W, 4) shape!
    latent_img_data = torch.sigmoid(latent_grid.detach()).cpu().numpy()
    latent_img_data = (latent_img_data * 255.0).astype(np.uint8)
    
    img = Image.fromarray(latent_img_data, mode="RGBA")
    png_path = os.path.join(args.output_dir, "latent_map.png")
    img.save(png_path)
    print("Saved latent map image to:", png_path)

if __name__ == "__main__":
    main()