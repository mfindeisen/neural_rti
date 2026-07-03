import os
import struct
import numpy as np

def read_ascii_line(f):
    line = bytearray()
    while True:
        b = f.read(1)
        if not b:
            break
        if b == b'\n':
            break
        line.extend(b)
    return line.decode('ascii').strip()

def load_hsh(filepath):
    """
    Loads an HSH (.rti) file and returns a dictionary with:
      - 'width': width of the image (int)
      - 'height': height of the image (int)
      - 'coeff_number': number of HSH coefficients (int)
      - 'bias': array of shape (coeff_number,) representing gmin
      - 'scale': array of shape (coeff_number,) representing gmax
      - 'coeffs': array of shape (H, W, 3, coeff_number) containing raw bytes (uint8)
    """
    with open(filepath, 'rb') as f:
        # Read lines skipping comments
        lines = []
        while len(lines) < 3:
            line = read_ascii_line(f)
            if not line.startswith('#') and len(line) > 0:
                lines.append(line)
        
        format_ver = lines[0]
        w, h, channels = map(int, lines[1].split())
        coeff_number, basis_type, element_size = map(int, lines[2].split())
        
        if basis_type != 2:
            raise ValueError(f"Unsupported basis type: {basis_type} (expected 2 for HSH)")
        
        # Read gmin and gmax float32 arrays
        gmin_bytes = f.read(coeff_number * 4)
        gmax_bytes = f.read(coeff_number * 4)
        
        gmin = np.frombuffer(gmin_bytes, dtype=np.float32)
        gmax = np.frombuffer(gmax_bytes, dtype=np.float32)
        
        # Read binary coefficient data
        data_bytes = f.read()
        expected_len = h * w * coeff_number * 3
        if len(data_bytes) < expected_len:
            raise ValueError(f"Truncated file: read {len(data_bytes)} bytes, expected {expected_len}")
        elif len(data_bytes) > expected_len:
            data_bytes = data_bytes[:expected_len]
            
        coeffs = np.frombuffer(data_bytes, dtype=np.uint8).reshape((h, w, 3, coeff_number))
        
        return {
            'width': w,
            'height': h,
            'coeff_number': coeff_number,
            'bias': gmin,
            'scale': gmax,
            'coeffs': coeffs
        }

if __name__ == "__main__":
    # Quick self-test
    path = "c:/Users/m/Projects/webRTIViewer/1921_7-DK-2_8256.rti"
    if os.path.exists(path):
        data = load_hsh(path)
        print("Successfully parsed HSH file:")
        print("Width:", data['width'])
        print("Height:", data['height'])
        print("Coefficients:", data['coeff_number'])
        print("Bias (gmin):", data['bias'])
        print("Scale (gmax):", data['scale'])
        print("Coefficients array shape:", data['coeffs'].shape)
    else:
        print("File not found for testing.")
