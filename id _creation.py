import os
import glob

# ====================================================
# 1️⃣ Base directory for all reference images
# ====================================================
BASE_REF_DIR = "/content/id_reference_images"

# List of known celebrity IDs (replace or load dynamically)
celebrity_ids = [
    "10173", "1158", "1499", "1757", "1852", "1964", "228", "2425",
    "2463", "2522", "2820", "2837", "2880", "3227", "3321", "3401",
    "3431", "3698", "3699", "3745", "3782", "4126", "4304", "447",
    "4561", "487", "5239", "5260", "6098", "619", "6568", "7282",
    "7904", "800", "8045", "8265", "8656", "8722", "8871", "8945",
    "8968", "9063", "9151", "9152", "9256", "9319"
]

# ====================================================
# 2️⃣ Create folder structure: one folder per ID
# ====================================================
os.makedirs(BASE_REF_DIR, exist_ok=True)

for cid in celebrity_ids:
    folder_path = os.path.join(BASE_REF_DIR, cid)
    os.makedirs(folder_path, exist_ok=True)
    print(f"[INFO] Created folder: {folder_path}")

print(f"\n[INFO] ✅ All {len(celebrity_ids)} ID folders created.")
print("[ACTION] Now manually place one JPEG per ID inside each folder.")
print("Example: /content/id_reference_images/10173/10173.jpg")

# ====================================================
# 3️⃣ Scan directories and build ID → image map
# ====================================================
id_db = {}
for folder in glob.glob(os.path.join(BASE_REF_DIR, "*")):
    if os.path.isdir(folder):
        id_num = os.path.basename(folder)
        img_files = glob.glob(os.path.join(folder, "*.jpg"))
        if img_files:
            id_db[id_num] = img_files[0]  # pick first jpg in that folder
        else:
            id_db[id_num] = None  # placeholder, no image yet

print(f"\n[INFO] Built ID reference map for {len(id_db)} IDs.")
filled = sum(1 for v in id_db.values() if v is not None)
print(f"[INFO] {filled} IDs already have images, {len(id_db)-filled} empty folders.")

# Example preview
print("\n[INFO] Sample entries:")
for k, v in list(id_db.items())[:5]:
    print(f"  {k} → {v}")
