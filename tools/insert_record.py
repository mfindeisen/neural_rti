import sqlite3
import datetime
import os

db_path = r"C:\Users\m\Projects\rtiDb\server\data\database.sqlite"

if not os.path.exists(db_path):
    print(f"Error: Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Insert the record
name = "Neural RTI - 1921_7-DK-2_8256"
description = "Neural RTI representation of 1921_7-DK-2_8256.rti, trained for 50 epochs on GPU. Evaluated full-resolution (8256x5504) training and validation lights at 31.91 dB PSNR. Compression: 6x (90.9 MB latent map + 13 KB weights JSON)."
date_str = datetime.datetime.now().isoformat()
tiff_url = "/static/uploads/neural_mona.tif"
output_type = "geotiff"
status = "done"
progress = 100
is_published = 1
direction = "ltr"

cursor.execute("""
    INSERT INTO records (name, description, date, tiff_url, output_type, status, progress, is_published, direction)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (name, description, date_str, tiff_url, output_type, status, progress, is_published, direction))

conn.commit()
record_id = cursor.lastrowid
print(f"Successfully inserted record into database with ID: {record_id}")
conn.close()
