# Setting up the benchmark on Ruche

## Getting the Code
The repository is private, so you need an SSH key to access it.

Generate an SSH key and add it to GitHub. For details, refer to this [tutorial](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

Clone the repository to your home directory **using SSH**, for example:

```shell
cd $HOME && git clone git@github.com:sonic160/HumanEval-Rel.git
```

## Setting up the Conda Environment

### Optional
Launch an interactive session on a CPU server (faster installation; still works on the frontend node):

```shell
srun --nodes=1 --time=00:30:00 -p cpu_short --pty /bin/bash
```

### Automatic Conda Setup
```shell
source $HOME/HumanEval-Rel/config_env.sh
```
You are ready to go!

---

### Manual Conda Setup
In this session:

```shell
module purge
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/11.8.0/gcc-11.2.0
# The following command installs every required package except flash attention
conda env create -n benchmark -f $HOME/HumanEval-Rel/benchmark_conda_env.yml
mkdir $WORKDIR/.cache && mkdir $WORKDIR/.cache/huggingface
# Activate the environment
source activate benchmark
```

Then, in the same session (after activating the conda environment), install flash attention. This step is done afterward to ensure that the correct CUDA version is used when building the wheel files.

### Option 1: Installation by Compilation (may take some time)
```shell
# Load a compatible compiler to build the flash-attn wheel
module load gcc/11.2.0/gcc-4.8.5   
python -m pip install flash-attn --no-build-isolation --no-cache-dir
# --no-build-isolation: recommended from flash-attention's GitHub
# --no-cache-dir: ensures the package is rebuilt
```

### Option 2: Installation via Precompiled Wheel (recommended)
To avoid compilation issues, you can use a precompiled version of flash-attention:

1. First, determine your environment parameters:
```shell
# Python version (to identify cp310, cp311, cp312, etc.)
python --version

# PyTorch version (to identify torch2.1, torch2.2, etc.)
python -c "import torch; print(torch.__version__)"

# CUDA version (to identify cu11, cu12, etc.)
python -c "import torch; print(torch.version.cuda)"

# Check the value of the CXX11_ABI flag (TRUE or FALSE)
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```

2. Download and install the corresponding wheel:
```shell
# Create a temporary folder
mkdir -p ~/flash_attn_temp && cd ~/flash_attn_temp

# Download the appropriate wheel 
# Replace the values based on your results (example for Python 3.10, PyTorch 2.1, CUDA 11.8, ABI=FALSE)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install the downloaded wheel
pip install flash_attn-2.7.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Return to the previous directory
cd -
```

You can find all available versions on the [flash-attention GitHub releases page](https://github.com/Dao-AILab/flash-attention/releases).

When everything is set up, you can launch the benchmark.