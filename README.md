# SpecLig: Energy-Guided Hierarchical Model for Target-Specific 3D Ligand Design

![cover](./assets/cover.png)

## :dna: Introduction

This is the official repository for our paper [SpecLig: Energy-Guided Hierarchical Model for Target-Specific 3D Ligand Design](https). We think specificity has been insufficiently addressed in the current field of structure-based ligand design, yet it becomes a critical barrier to the translation from computational design to clinical application. SpecLig enables the simultaneous design of small molecules and peptides with high affinity and specificity against target proteins, as the physics-based interactions and local geometric features remain consistent. Although the present framework may still be oversimplified for practical applications, we are rigorously refining it through the integration of additional data and functional modules. Thank you for your interest in our work!

## :mag: Quick Links

- [Setup](#rocket-setup)
  - [Environment](#environment)
  - [Trained Weights](#trained-weights)
- [Reproduction of Paper Experiments](#page_facing_up-reproduction-of-paper-experiments)
  - [Additional Dependencies](#additional-dependencies)
  - [Datasets](#datasets)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Contact](#bulb-contact)
- [Reference](#reference)

## :rocket: Setup

### Environment

We have prepared conda environment configurations for **cuda 11.7 + pytorch 1.13.1** (`env_cuda117.yaml`) and **cuda 12.1 + pytorch 2.1.2** (`env_cuda121.yaml`). For example, you can create the environment by:

```bash
conda env create -f env_cuda117.yaml
```

Remember to activate the environment before running the codes:

```bash
conda activate SpecLig
```

### Trained Weights

We have uploaded the trained checkpoint at the [release page](https://cloud.tsinghua.edu.cn/d/f26ee0ca563d4150a585/). Please download it and put it under `./checkpoints/model.ckpt`.

## :page_facing_up: Reproduction of Paper Experiments

### Additional Dependencies

#### PyRosetta

PyRosetta is used to calculate interface energy of generated peptides. Please follow the official instruction [here](https://www.pyrosetta.org/downloads) to install it.

#### CBGBench

CBGBench is used to evaluate designed small molecules comprehensively. Please follow its [official repository](https://github.com/EDAPINENUT/CBGBench) to prepare the codes and environment. We recommend building another environment specifically for CBGBench to separate the dependencies from the environment of SpecLig.

### Datasets

**Throughout the instructions, we suppose all the datasets are downloaded below `./datasets`.**

Please refer to [`README.md`](./datasets/README.md) in the `datasets` folder.

### Training

Training of the full SpecLig requires 8 A800 GPUs with 80G memmory each. The process includes training an all-atom variational encoder, and a block-level latent diffusion model, which commonly takes about 2-3 days.

```bash
python train.py --gpus 1 --config ./configs/IterAE/train.yaml

nohup python -u train.py --gpus 4  --config ./configs/LDM/train.yaml  --trainer.config.save_dir=./ckpts/speclig/LDM --model.autoencoder_ckpt=./checkpoints/vae.ckpt > diff.log 2>&1 &
```

### Inference

The following commands generate 100 candidates for each target in the test sets, which are LNR and CrossDocked test set for peptide and small molecule, respectively.

```bash
# peptide
nohup python -u generate.py --config configs/test/test_pep.yaml --ckpt checkpoints/model.ckpt --gpu 5 --save_dir ./results/pep > pep.log 2>&1 &
# small molecule
nohup python -u generate.py --config configs/test/test_mol.yaml --ckpt checkpoints/model.ckpt --gpu 6 --save_dir ./results/mol > mol.log 2>&1 &
```

### Evaluation

:question: Due to the non-deterministic behavior of `torch.scatter`, the reproduced results might not be exactly the same as those reported in the paper, but should be very close to them.

Please revise the path to CBGBench and CrossDocked2020 in `./scripts/metrics/mol.sh`:

```bash
# Line 5 and line 6 in ./scripts/metrics/mol.sh
CBGBENCH_REPO=/path/to/CBGBench
DATA_DIR=/path/to/CrossDocked/crossdocked_pocket10
```

The evaluation scripts are as follows. Note that the evaluation process is CPU-intensive. During our experiments, each of them requires running for several hours on 96 cpu cores.

```bash
# peptide
python -m scripts.metrics.peptide --results ./results/pep/results.jsonl --num_workers 96
# small molecule
conda activate cbgbench # use its own environment
nohup bash scripts/metrics/mol.sh ./results/mol/candidates > eval_mol_.log 2>&1 &
```

## :bulb: Contact

Thank you for your interest in our work!

Please feel free to ask about any questions about the algorithms, codes, as well as problems encountered in running them so that we can make it clearer and better. You can either create an issue in the github repo or contact us at zpd24@mails.tsinghua.edu.cn.

```
