import os
import re
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PIL import Image
from mail import send_professor_email
from typing import Optional   # ✅ <-- Add this

# ======================================================
# CONFIG
# ======================================================
MODELS = {
    "single": [
        "models/best_face_yolo.pt",
        "models/yolov8n-face.pt"
    ],
    "group": [
        "models/best_face_yolo.pt",
    ]
}

ID_TO_NAME_CSV = "id_to_name.csv"
ID_REFERENCE_DIR = "id_reference_images"
PROFESSOR_EMAIL = "pexito2569@bdnets.com"


# ======================================================
# HELPERS
# ======================================================
def parse_real_id(label: str) -> Optional[str]:   # ✅ fixed type hint
    """
    Try to extract the real celebrity ID from a model class label.
    Works for labels like '<4126>', '4126', 'id_4126', 'ID:4126'.
    Returns the first/longest digit run if present, else None.
    """
    if label is None:
        return None
    s = str(label).strip()
    if s.isdigit():
        return s
    nums = re.findall(r"\d+", s)
    if not nums:
        return None
    return max(nums, key=len)



# ======================================================
# LOAD REFERENCE DATA
# ======================================================
def load_reference_data(ref_dir=ID_REFERENCE_DIR):
    """Loads reference images for each known ID."""
    print(f"[INFO] Loading reference data from '{ref_dir}'...")
    ref_data = {}

    if not os.path.exists(ref_dir):
        print(f"[WARN] Reference directory not found: {ref_dir}")
        return ref_data

    # Case 1: files directly inside
    for f in os.listdir(ref_dir):
        p = os.path.join(ref_dir, f)
        if os.path.isfile(p) and f.lower().endswith((".jpg", ".jpeg", ".png")):
            id_ = os.path.splitext(f)[0]
            ref_data[id_] = p

    # Case 2: subfolders
    for sub in os.listdir(ref_dir):
        subp = os.path.join(ref_dir, sub)
        if os.path.isdir(subp):
            imgs = [os.path.join(subp, x) for x in os.listdir(subp)
                    if x.lower().endswith((".jpg", ".jpeg", ".png"))]
            if imgs:
                ref_data[sub] = imgs[0]

    print(f"[INFO] ✅ Loaded {len(ref_data)} reference entries.")
    if len(ref_data) > 0:
        example = list(ref_data.items())[:3]
        print(f"[DEBUG] Example references: {example}")
    return ref_data


# ======================================================
# LOAD ID→NAME MAP
# ======================================================
def load_id_to_name(csv_path=ID_TO_NAME_CSV):
    """Loads ID→Name mapping from CSV."""
    print(f"[INFO] Loading ID→Name mapping from '{csv_path}'...")
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        id_to_name = {}
        for _, row in df.iterrows():
            sid = str(row["Celebrity Choice/ID"]).strip()
            name = str(row["Student Name"]).strip()
            id_to_name[sid] = name
        print(f"[INFO] ✅ Loaded {len(id_to_name)} mappings from CSV.")
        if len(id_to_name) > 0:
            print(f"[DEBUG] Example mappings: {list(id_to_name.items())[:5]}")
        return id_to_name
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return {}


# ======================================================
# LOAD YOLO MODELS
# ======================================================
def load_models():
    """
    Load YOLO models and capture their class-name maps (model.names).
    Returns:
      {
        "single": [(model_file, model_obj, names_map), ...],
        "group":  [(model_file, model_obj, names_map), ...]
      }
    """
    print("[INFO] Loading YOLO models into memory...")
    models = {}
    for mode, paths in MODELS.items():
        models[mode] = []
        for path in paths:
            if not os.path.exists(path):
                print(f"[WARN] Model not found: {path}")
                continue
            print(f"[DEBUG] Loading {mode} model: {path}")
            model = YOLO(path)
            names_map = getattr(model, "names", {}) or {}
            # Show a peek at the names map
            if isinstance(names_map, dict) and len(names_map):
                sample = list(names_map.items())[:5]
                print(f"[DEBUG] names_map for {os.path.basename(path)}: {sample}")
            else:
                print(f"[DEBUG] names_map for {os.path.basename(path)} is empty/unknown.")
            models[mode].append((os.path.basename(path), model, names_map))
            print(f"[DEBUG] ✅ Loaded: {os.path.basename(path)}")

    total = sum(len(v) for v in models.values())
    print(f"[INFO] ✅ Total models loaded: {total}")
    return models


# ======================================================
# DETECT IMAGE TYPE
# ======================================================
def detect_image_type(image):
    """Rough heuristic: group photo if aspect ratio > 1.3."""
    w, h = image.size
    ratio = w / h
    print(f"[DEBUG] Image size: {w}x{h} (ratio={ratio:.2f})")
    image_type = "group" if ratio > 1.3 else "single"
    print(f"[INFO] Detected image type: {image_type}")
    return image_type


# ======================================================
# MAIN INFERENCE PIPELINE
# ======================================================
def identify_faces(image, id_reference):
    """
    Full inference and postprocessing pipeline.
    Runs multiple YOLO models, selects best, returns annotated image + table,
    and sends attendance email to the professor.
    """
    from PIL import Image as PILImage

    # Convert string path → PIL Image if needed
    if isinstance(image, str):
        print(f"[DEBUG] Received image path instead of PIL object: {image}")
        image = PILImage.open(image)

    print("[INFO] Initializing inference pipeline...")
    models = load_models()
    id_to_name = load_id_to_name(ID_TO_NAME_CSV)

    # ----------------------------------------------------
    # Detect if group or single image
    # ----------------------------------------------------
    image_type = detect_image_type(image)
    available_models = models.get(image_type, [])
    if not available_models:
        print(f"[ERROR] ❌ No models loaded for image type '{image_type}'.")
        return image, pd.DataFrame()

    print(f"[INFO] 🧠 Running {len(available_models)} {image_type} models for comparison...")
    img_np = np.array(image)
    model_results = {}

    # ----------------------------------------------------
    # Run all models sequentially
    # ----------------------------------------------------
    for model_name, model, names_map in available_models:
        print(f"\n[INFO] 🔍 Running inference using model '{model_name}'...")
        try:
            results = model.predict(img_np, conf=0.5, verbose=False)
        except Exception as e:
            print(f"[ERROR] Failed to predict with {model_name}: {e}")
            continue

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print(f"[WARN] ❌ No detections found by {model_name}.")
            continue

        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        print(f"[DEBUG] Detected {len(cls)} faces | Raw classes: {cls.tolist()} | Confidences: {np.round(conf, 3)}")

        # Translate class index -> label -> real celeb id
        out_ids, out_names, out_refs = [], [], []
        debug_rows = []
        for cid, score in zip(cls, conf):
            label = names_map.get(int(cid), str(cid)) if isinstance(names_map, dict) else str(cid)
            real_id = parse_real_id(label)
            if real_id is None:
                # Fall back to class idx (likely not a custom-ID model)
                real_id = str(cid)

            # Lookup name and reference
            name = id_to_name.get(real_id, f"Unknown (ID {real_id})")
            refp = id_reference.get(real_id, "❌ Not Found")

            out_ids.append(real_id)
            out_names.append(name)
            out_refs.append(refp)
            debug_rows.append((cid, label, real_id, name, score))

        # Debug dump per detection row
        for row in debug_rows[:10]:
            _cid, _label, _rid, _name, _sc = row
            print(f"[DEBUG] map: cls={_cid} → label='{_label}' → id={_rid} → name='{_name}' (conf={_sc:.3f})")
        if len(debug_rows) > 10:
            print(f"[DEBUG] ... {len(debug_rows)-10} more rows omitted")

        df = pd.DataFrame({
            "ID": out_ids,
            "Celebrity Name": out_names,
            "Confidence": np.round(conf, 3),
            "Reference Image": out_refs
        })

        mean_conf = float(np.mean(conf)) if len(conf) else 0.0
        print(f"[DEBUG] Mean confidence for {model_name}: {mean_conf:.3f}")
        model_results[model_name] = {"df": df, "mean_conf": mean_conf, "results": results}

    # ----------------------------------------------------
    # Select Best Model
    # ----------------------------------------------------
    if not model_results:
        print("[WARN] ❌ No detections from any model.")
        return image, pd.DataFrame()

    best_name, best_info = max(model_results.items(), key=lambda x: x[1]["mean_conf"])
    print(f"[INFO] ✅ Best model selected: {best_name} (mean_conf={best_info['mean_conf']:.3f})")

    # ----------------------------------------------------
    # Postprocess
    # ----------------------------------------------------
    annotated = best_info["results"][0].plot()
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    df = best_info["df"]

    print(f"[DEBUG] Detection results table:\n{df}")
    recognized_people = list(zip(df["ID"], df["Celebrity Name"]))
    print(f"[DEBUG] Recognized people list: {recognized_people}")

    # ----------------------------------------------------
    # Send Professor Email
    # ----------------------------------------------------
    if recognized_people:
        print(f"[INFO] ✉️ Sending attendance summary to {PROFESSOR_EMAIL}...")
        mail_status = send_professor_email(recognized_people, PROFESSOR_EMAIL)
        if mail_status:
            print("[INFO] ✅ Professor email sent successfully.")
        else:
            print("[WARN] ⚠️ Professor email failed to send.")
    else:
        print("[WARN] ⚠️ No recognized faces; skipping email notification.")

    print("[INFO] ✅ Inference pipeline completed successfully.")
    return annotated_pil, df
