import os
import torch


def setup_device():
    """Configure the correct GPU device based on SLURM allocation"""
    # Check for SLURM GPU allocation
    slurm_gpus = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")

    # Determine if CUDA is available
    cuda_available = torch.cuda.is_available()

    if cuda_available and slurm_gpus:
        print(f"SLURM assigned GPUs: {slurm_gpus}")
        # Use the SLURM assigned GPUs
        if "," in slurm_gpus:
            # Multiple GPUs case
            gpu_ids = slurm_gpus.split(",")
        elif "-" in slurm_gpus:
            # Range format (e.g., "0-3")
            start, end = map(int, slurm_gpus.split("-"))
            gpu_ids = [str(i) for i in range(start, end + 1)]
        else:
            # Single GPU
            gpu_ids = [slurm_gpus]

        # Set environment variable for PyTorch
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        print(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")
        return torch.device("cuda")
    else:
        return torch.device("cuda") if cuda_available else torch.device("cpu")
