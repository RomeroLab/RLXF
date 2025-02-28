# RLXF  
Consolidated repository to perform RLXF.  

Perform the following steps to functionally align the protein language model (pLM) **ESM-2**.

## Key Terminology  
- **SFT**: Supervised Fine-tuning  
- **PPO**: Proximal Policy Optimization

## Step 0: Pre-process Sequence-Function Dataset  

Ensure the sequence-function dataset file is correctly formatted before proceeding.

## **File Naming**  
- The dataset file should be named:  
  **`SeqFxnDataset.pkl`**  

## **Column Naming**  
- The column containing **amino acid sequences** must be named:  
  **`sequence`**  
- The column containing **functional scores** must be named:  
  **`functional_score`**  

## Step 1: Train reward model
We train an ensemble of multi-layer perceptrons to predict the log fluoresence of CreiLOV variants in a DMS dataset. Our repository is setup to train on a sequence-function dataset file (SeqFxnDataset.pkl) with a sequence and functional_score column.

```python3 Training_Ensemble_of_reward_models.py > Training_Ensemble_of_reward_models.out```

Files generated:

## Step 2: Perform simulated annealing
Generate a small, high quality synthetic sequence dataset for SFT

```python3 script.py > script.out```

Files generated:

## Step 3: SFT
Supervise finetune pLM

```python3 script.py > script.out```

Files generated:

## Step 4: Perform PPO
Align SFT-pLM with proximal policy optimization

```python3 script.py > script.out```

Files generated:

## Step 5: Generate designs
Generate designs to characterize

```python3 script.py > script.out```

Files generated:

