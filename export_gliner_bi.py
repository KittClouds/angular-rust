import sys
import subprocess

def install_deps():
    print("Checking dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gliner==0.2.8", "onnx", "onnxruntime"])

try:
    import gliner
except ImportError:
    install_deps()

from gliner import GLiNER

def main():
    model_id = "knowledgator/gliner-bi-base-v2.0"
    local_dir = "gliner-bi-onnx"

    print(f"Loading {model_id}...")
    model = GLiNER.from_pretrained(model_id)

    print(f"Saving to local directory {local_dir}...")
    model.save_pretrained(local_dir)

    print("Initializing ONNX exporter...")
    gliner_model = GLiNER.from_pretrained(local_dir, load_tokenizer=True)

    print("Triggering ONNX export (this might take a minute)...")
    gliner_model.export_to_onnx(
        save_dir=local_dir,
        onnx_filename="model.onnx", 
        quantized_filename="model_quantized.onnx",
        quantize=True,
        opset=18
    )

    print(f"Export complete! Check the {local_dir} directory.")

if __name__ == "__main__":
    main()