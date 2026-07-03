import os
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
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
    import argparse
    parser = argparse.ArgumentParser(description="Neural RTI Evaluation Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to original input .rti file")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained decoder_weights.json")
    parser.add_argument("--latent", type=str, required=True, help="Path to trained latent_map.png")
    args = parser.parse_args()
    
    rti_path = args.input
    weights_path = args.weights
    latent_path = args.latent
    
    print("Checking if files exist...")
    if not os.path.exists(rti_path):
        print(f"Error: RTI file not found at {rti_path}")
        return
    if not os.path.exists(weights_path):
        print(f"Error: weights not found at {weights_path}")
        return
    if not os.path.exists(latent_path):
        print(f"Error: latent map not found at {latent_path}")
        return
        
    print("Loading HSH file...")
    data = load_hsh(rti_path)
    coeffs_np, bias_np, scale_np = data['coeffs'], data['bias'], data['scale']
    H, W, C, K = coeffs_np.shape
    print(f"RTI original dimensions: {W}x{H}, channels: {C}, coeffs: {K}")
    
    print("Loading latent map...")
    latent_img = Image.open(latent_path).convert("RGBA")
    latent_np = np.array(latent_img, dtype=np.float32) / 255.0
    latent_H, latent_W, latent_D = latent_np.shape
    print(f"Latent map dimensions: {latent_W}x{latent_H}, channels: {latent_D}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if latent_H != H or latent_W != W:
        print(f"Warning: Latent map dimensions ({latent_W}x{latent_H}) do not match original RTI dimensions ({W}x{H})!")
        print("Downsampling coefficients to match latent map dimensions...")
        coeffs_temp = torch.from_numpy(coeffs_np.copy()).float()
        # Shape prep: (C*K, H, W)
        coeffs_flat_temp = coeffs_temp.permute(2, 3, 0, 1).reshape(C*K, H, W).unsqueeze(0)
        coeffs_resized = nn.functional.interpolate(coeffs_flat_temp, size=(latent_H, latent_W), mode='bilinear', align_corners=False)
        coeffs = coeffs_resized.squeeze(0).reshape(C, K, latent_H, latent_W).permute(2, 3, 0, 1).byte().to(device)
        H, W = latent_H, latent_W
    else:
        coeffs = torch.from_numpy(coeffs_np.copy()).to(device)
        
    print("Loading decoder...")
    decoder = load_decoder_from_json(weights_path)
    decoder = decoder.to(device)
    
    bias = torch.from_numpy(bias_np.copy()).to(device)
    scale = torch.from_numpy(scale_np.copy()).to(device)
    latent = torch.from_numpy(latent_np.copy()).to(device)

    
    # Let's evaluate reconstruction quality on train lights
    train_lights = sample_hemisphere_lights(64).to(device)
    
    # Select 10 random lights on the hemisphere for generalization test
    np.random.seed(42)
    val_lights = []
    for _ in range(10):
        # random direction on hemisphere (z > 0)
        phi = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(0.05, 1.0)
        r = np.sqrt(1.0 - z*z)
        x = np.cos(phi) * r
        y = np.sin(phi) * r
        val_lights.append([x, y, z])
    val_lights = torch.tensor(val_lights, dtype=torch.float32, device=device)
    
    # To avoid OOM, process in blocks or pixels if the image is very large.
    # WxH is 8256. Wait, let's see. The filename contains "8256".
    # Wait, the file size is 545MB. 
    # Let's check what the size is. If it's very large, we should sample a subset of pixels for evaluation.
    # Let's do 1,000,000 random pixels to get a highly accurate statistic without OOM.
    num_samples = min(1000000, H * W)
    pixel_indices = np.random.choice(H * W, num_samples, replace=False)
    
    coeffs_flat = coeffs.view(-1, C, K)[pixel_indices]
    latent_flat = latent.view(-1, latent_D)[pixel_indices]
    
    print(f"Evaluating {num_samples} sampled pixels...")
    
    # 1. Evaluation on training lights
    train_mses = []
    for i in range(len(train_lights)):
        l_dir = train_lights[i:i+1] # (1, 3)
        with torch.no_grad():
            target_rgb = evaluate_hsh_torch(coeffs_flat, bias, scale, l_dir).squeeze(0)
            
            l_dir_expand = l_dir.expand(num_samples, 3)
            pred_rgb = decoder(latent_flat, l_dir_expand)
            
            mse = torch.mean((pred_rgb - target_rgb) ** 2).item()
            train_mses.append(mse)
            
    mean_train_mse = np.mean(train_mses)
    mean_train_psnr = 20 * np.log10(1.0 / np.sqrt(mean_train_mse))
    
    # 2. Evaluation on validation (unseen) lights
    val_mses = []
    for i in range(len(val_lights)):
        l_dir = val_lights[i:i+1]
        with torch.no_grad():
            target_rgb = evaluate_hsh_torch(coeffs_flat, bias, scale, l_dir).squeeze(0)
            
            l_dir_expand = l_dir.expand(num_samples, 3)
            pred_rgb = decoder(latent_flat, l_dir_expand)
            
            mse = torch.mean((pred_rgb - target_rgb) ** 2).item()
            val_mses.append(mse)
            
    mean_val_mse = np.mean(val_mses)
    mean_val_psnr = 20 * np.log10(1.0 / np.sqrt(mean_val_mse))
    
    print("\n--- Evaluation Results ---")
    print(f"Training MSE: {mean_train_mse:.7f}")
    print(f"Training PSNR: {mean_train_psnr:.2f} dB")
    print(f"Validation (Unseen lights) MSE: {mean_val_mse:.7f}")
    print(f"Validation (Unseen lights) PSNR: {mean_val_psnr:.2f} dB")
    
    # Check max difference
    print(f"Max train MSE across lights: {np.max(train_mses):.7f}")
    print(f"Min train MSE across lights: {np.min(train_mses):.7f}")

if __name__ == "__main__":
    main()
