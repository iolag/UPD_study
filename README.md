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

To be able to use [Weights & Biases](https://wandb.ai) for logging:

a) Sign up for a free account at https://wandb.ai/site and then login to your wandb account.
b) Find your API key here: https://wandb.ai/authorize and login to the wandb library on your machine by running:
```bash
wandb login
```
