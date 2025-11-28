# services/storage/backup.py
import os
import shutil
from datetime import datetime

def backup_files():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    files_to_backup = ["faiss.index", "id_map.json", "documents.json"]
    backup_dir = f"backup_{timestamp}"

    os.makedirs(backup_dir, exist_ok=True)

    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy(file, os.path.join(backup_dir, file))
            print(f"Backed up {file} → {backup_dir}/{file}")
        else:
            print(f"Skipping {file} (not found).")

    print(f"\nBackup complete. Directory created: {backup_dir}")

if __name__ == "__main__":
    backup_files()
