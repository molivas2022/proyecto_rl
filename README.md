# MultiAgent MetaDrive Experiments

This project provides a implementation of MAPPO and IPPO for training autonomous driving agents in the MetaDrive environment. 

## Setup Instructions

### Prerequisites

- Conda
- PyTorch (install based on your CUDA configuration)

### Installation

1.  Clone the repository:

```bash
git clone <repository_url>
cd <repository_directory>
```

2.  Create the environment with conda:

```bash
conda env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate proyecto-rl
```


### Running Locally

1.  Configure the experiment in `experiments.yml`.
2.  Run the training script:

```bash
python train.py
```

This will start the training process using the specified configuration.  You can monitor the training progress using TensorBoard or the Ray Tune web UI.
