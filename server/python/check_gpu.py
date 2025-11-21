"""
GPU Detection and Information Script
Run this to check if PyTorch can detect your GPU
"""
import torch

print("=" * 60)
print("PyTorch GPU Detection")
print("=" * 60)

# Check CUDA availability
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Multi-Processors: {props.multi_processor_count}")
    
    # Test GPU memory allocation
    print("\n" + "=" * 60)
    print("Testing GPU Memory Allocation...")
    print("=" * 60)
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = x @ y
        print("✓ Successfully allocated and performed operation on GPU")
        print(f"✓ Current GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"✓ Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
        # Cleanup
        del x, y, z
        torch.cuda.empty_cache()
        print("✓ GPU memory cleared")
    except Exception as e:
        print(f"✗ Error testing GPU: {e}")
else:
    print("\n⚠ No GPU detected!")
    print("\nPossible reasons:")
    print("1. No NVIDIA GPU installed")
    print("2. CUDA drivers not installed")
    print("3. PyTorch CPU-only version installed")
    print("\nTo install PyTorch with CUDA support, visit:")
    print("https://pytorch.org/get-started/locally/")
    
    # CPU info
    print(f"\n✓ CPU will be used for training")
    print(f"✓ Number of CPU threads: {torch.get_num_threads()}")

print("\n" + "=" * 60)
print("Detection Complete")
print("=" * 60)
