import sys
import torch
import torchvision

def main():
    print("=" * 50)
    print(" PyTorch & Hardware Environment Verification")
    print("=" * 50)
    
    python_ver = sys.version.split()[0]
    print(f"Python:          {python_ver}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"Torchvision:     {torchvision.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available:  {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version:    {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU Count:       {device_count}")
        for i in range(device_count):
            print(f"GPU {i}:           {torch.cuda.get_device_name(i)}")
        print("\nStatus: GPU Acceleration active. Style transfer will use CUDA.")
    else:
        print("CUDA version:    N/A")
        print("GPU:             None (Using CPU)")
        print("\nStatus: CUDA is not available. The application will run on CPU.")
        print("Neural Style Transfer inference will execute reliably using CPU mode.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
