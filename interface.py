import gradio as gr
from PIL import Image
from inference import identify_faces, load_reference_data

# ======================================================
# Load Reference Data Once
# ======================================================
print("[INFO] Loading reference data...")
REFERENCE_DIR = "id_reference_images"
id_reference = load_reference_data(REFERENCE_DIR)
print(f"[INFO] Loaded reference data for {len(id_reference)} IDs.")


# ======================================================
# Gradio Processing Function
# ======================================================
def process_image(image: Image.Image):
    if image is None:
        return None, "❌ Please upload an image."

    print("\n[INFO] 🖼️ Processing new upload...")
    annotated_img, results_table = identify_faces(image, id_reference)
    return annotated_img, results_table


# ======================================================
# Gradio Interface
# ======================================================
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="📸 Upload an Image (Single or Group)"),
    outputs=[
        gr.Image(label="🧠 Annotated Detection (Best Model Output)"),
        gr.Dataframe(
            headers=["ID", "Celebrity Name", "Confidence", "Reference Image"],
            label="Detection Results"
        )
    ],
    title="🎭 Multi-Model Celebrity Identification System",
    description=(
        "Upload a **single or group photo**.\n\n"
        "The system will:\n"
        "1️⃣ Detect faces using multiple YOLO models\n"
        "2️⃣ Pick the highest-confidence model\n"
        "3️⃣ Map IDs → Names and Reference photos\n"
        "4️⃣ Email the attendance summary to your Professor.\n\n"
        "**Output:** Annotated image + detection table"
    ),
    allow_flagging="never",
    api_name=False
)


def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    print("[INFO] 🚀 Launching Gradio Interface...")
    iface.launch(share=True, debug=True)
