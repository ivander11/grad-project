import torch
try:
    ckpt = torch.load('weights/best.pt', map_location='cpu')
    print("✅ Weights are healthy.")
except Exception as e:
    print(f"❌ Weights are still corrupted: {e}")