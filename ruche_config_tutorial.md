# Setting up the benchmark on Ruche 

## Getting the code
The repository we work with is private, you need an ssh key for ruche to access 

Generate an ssh key and add to github: here is a [tutorial](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

clone the repository to your home **using SSH** ie

```shell
cd $HOME && git clone git@github.com:sonic160/HumanEval-Rel.git
````

## Setting Conda environment

### Optional
 Launch an interactive session on a cpu server (faster installation, still works on the frontend node)

```shell
srun --nodes=1 --time=00:30:00 -p cpu_short --pty /bin/bash
```
### Automatic conda set up
```shell
source $HOME/HumanEval-Rel/config_env.sh
```
you are ready to go : )

---

### Manual conda set up
In this session :

```shell
module purge
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/11.8.0/gcc-11.2.0
#the following command installs every required pacakges except flash attention
conda env create -n benchmark -f $HOME/HumanEval-Rel/benchmark_conda_env.yml
mkdir $WORKDIR/.cache && mkdir $WORKDIR/.cache/huggingface
```
Then, in the same session (you need to be in the conda env) install flash attention, this is done after to ensure that it get the correct CUDA version when building the wheel files.
```shell
python -m pip install flash-attn --no-build-isolation --no-cache-dir
# --no-build-isolation: recommended from flash attention's github
# --no-cache-dir: makes sure the package is rebuilt 
```


When everything is done you can launch the benchmark using