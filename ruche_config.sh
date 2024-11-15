module purge
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/11.8.0/gcc-11.2.0
conda env create -n benchmark -f $HOME/HumanEval-Rel/benchmark_conda_env.yml
python -m pip install flash-attn --no-build-isolation --no-cache-dir
mkdir $WORKDIR/.cache && mkdir $WORKDIR/.cache/huggingface