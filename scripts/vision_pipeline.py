import torch
import time
import numpy as np
from ultralytics import RTDETR
from mobile_sam import sam_model_registry, SamPredictor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DETR_WEIGHTS = 'weights/best.pt'
SAM_WEIGHTS = 'weights/mobile_sam.pt'

def test_true_vram_load():
    print(f"Hardware Target: {DEVICE}")
    if DEVICE == 'cpu':
        print("CRITICAL WARNING: PyTorch is not using the GPU.")
    
    print("Loading models...")
    detr_model = RTDETR(DETR_WEIGHTS)
    
    mobile_sam = sam_model_registry["vit_t"](checkpoint=SAM_WEIGHTS)
    mobile_sam.to(device=DEVICE)
    mobile_sam.eval()
    sam_predictor = SamPredictor(mobile_sam)
    
    print("Executing warm-up inference to allocate tensor buffers...")
    # Create a blank 640x640 RGB image
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Force RT-DETR forward pass
    detr_model.predict(dummy_image, conf=0.5, verbose=False)
    
    # Force MobileSAM forward pass
    sam_predictor.set_image(dummy_image)
    dummy_box = np.array([10, 10, 100, 100])
    sam_predictor.predict(box=dummy_box, multimask_output=False)
    
    print("\n✅ SUCCESS: Full inference completed.")
    print("Holding models in memory for 15 seconds. Check nvidia-smi NOW.")
    time.sleep(15)
    print("Test complete. VRAM released.")

if __name__ == "__main__":
    test_true_vram_load()