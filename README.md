# UAD_study

# Usage

Download this repository by running

```bash
git clone https://github.com/iolag/UAD_study/
```

in your terminal.

## Environment

Create and activate the Anaconda environment:

```bash
conda env create -f environment.yml
conda activate upd
```

Additionally, you need to install the repository as a package:

```bash
python3 -m pip install --editable .
```

To be able to use [Weights & Biases](https://wandb.ai) for logging, set the environment variables $WANDBNAME and $WANDBPROJECT.
