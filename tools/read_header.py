import os
import struct

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

def print_header(filepath):
    print("Reading header of:", filepath)
    with open(filepath, 'rb') as f:
        # Read lines skipping comments
        lines = []
        while len(lines) < 3:
            line = read_ascii_line(f)
            if not line.startswith('#') and len(line) > 0:
                lines.append(line)
        
        print("Format line:", lines[0])
        print("Size line:", lines[1])
        print("Basis line:", lines[2])
        
        # Read gmin and gmax
        w, h, channels = map(int, lines[1].split())
        coeff_number, basis_type, element_size = map(int, lines[2].split())
        print(f"w: {w}, h: {h}, channels: {channels}")
        print(f"coeff_number: {coeff_number}, basis_type: {basis_type}, element_size: {element_size}")

if __name__ == "__main__":
    import sys
    path = "c:/Users/m/Projects/webRTIViewer/1921_7-DK-2_8256.rti"
    if os.path.exists(path):
        print_header(path)
    else:
        print("File not found:", path)
