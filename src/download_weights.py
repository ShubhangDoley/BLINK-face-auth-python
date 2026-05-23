import os
import urllib.request

def download_face_weights():
    """
    Downloads the pre-trained SFace Face Recognition ONNX model file 
    from OpenCV's Hugging Face Zoo and saves it to the models/ directory.
    This model has the standard (1, 3, 112, 112) input and generates 
    128D unit-normalized feature embedding vectors, serving as a 
    production-grade, 99%+ accurate offline replacement.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Target path
    onnx_filename = "mobilefacenet.onnx"
    target_path = os.path.join(models_dir, onnx_filename)
    
    # Standard, verified face recognition ONNX model URL (OpenCV Zoo SFace)
    # Size: ~36MB
    url = "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx?download=true"
    
    print("\n" + "="*50)
    print("BLINK - NEURAL NETWORK WEIGHT DOWNLOADER")
    print("="*50)
    print(f"[INFO] Target Path: {target_path}")
    print("[INFO] Downloading SFace Face Recognition ONNX Model (~36MB)...")
    print("[INFO] This may take a moment depending on your internet connection.")
    
    try:
        # Progress callback hook
        def progress_callback(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100.0, (downloaded / total_size) * 100.0) if total_size > 0 else 0
            print(f"\r>>> Progress: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)", end="")

        urllib.request.urlretrieve(url, target_path, reporthook=progress_callback)
        print("\n" + "="*50)
        print(f"[SUCCESS] Face recognition weights successfully saved to: {target_path}")
        print("[INFO] Restart your src/main.py stream to activate production-grade neural inference!")
        print("="*50 + "\n")
        
    except Exception as e:
        print("\n" + "!"*50)
        print(f"[ERROR] Failed to download model weights: {e}")
        print("[INFO] Please verify your internet connection or download manually from:")
        print("https://huggingface.co/opencv/face_recognition_sface")
        print("!"*50 + "\n")

if __name__ == "__main__":
    download_face_weights()
