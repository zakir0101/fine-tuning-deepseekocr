import os
from pathlib import Path
import shutil

temp_igcse = Path("/home/zakir/Documents/tmp_files/IGCSE_DATA/")
IGCSE_HOME = "/mnt/wsl/Drive/IGCSE-NEW/"

json_files = list(temp_igcse.glob("*/ft5-deepseekocr/*/v1.json"))
for file in json_files:
    sub, backend, exam, filename = str(file).split("/")[-4:]
    output_dir = os.path.join(IGCSE_HOME, sub, backend, exam)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    shutil.copy(file, output_path)
    print("file saved to ", output_path)
