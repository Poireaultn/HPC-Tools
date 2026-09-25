# Deliverable 1: BASELINE

This directory contains the single-GPU baseline implementation for fine-tuning `google-bert/bert-base-uncased` on the `rajpurkar/squad` dataset on the CESGA FinisTerrae III (FT3) supercomputer.

## Environment Setup (FT3)
To reproduce the virtual environment on FT3 compatible with the A100 CUDA drivers:

```bash
module load cesga/2022
module load python/3.10.8
python3 -m venv $STORE/mypython
source $STORE/mypython/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets
```

## File Structure
* `train_bert.py`: PyTorch training script using `BertForQuestionAnswering`, an optimized `DataLoader` (`batch_size=16`, `num_workers=4`, `pin_memory=True`), and accurate GPU timing via `torch.cuda.synchronize()`.
* `run_baseline.slurm`: SLURM batch script allocating 1 Node, 1 NVIDIA A100 GPU, 32 CPU cores, and 64 GB of RAM.

## Execution
Submit the job to SLURM from the `Baseline` directory:

```bash
sbatch run_baseline.slurm
```

## Baseline Results
* **Dataset subset**: 5,000 training examples (`max_length=384`)
* **Training configuration**: 2 epochs, batch size = 16, AdamW (`lr=3e-5`)
* **Hardware**: 1x NVIDIA A100 GPU (32 CPU cores, 64 GB RAM)
* **Total GPU training time**: **159.11 seconds**