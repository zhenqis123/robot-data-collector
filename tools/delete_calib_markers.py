import json
from pathlib import Path
des_path = "data/captures/2025-12-25"
des_path = Path(des_path)
file_list = des_path.glob("*/postprocess_markers.json")
print(file_list)