import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    )

    print("VRAM:", round(gpu_memory, 2), "GB")
else:
    print("CUDA is not available. Training will need CPU or CUDA setup.")