# RLXF
Consolidated repository to perform RLXF

Notes:
SFT: Supervised finetuning
PPO: Proximal policy optimization

### Step 0: Pre-process sequence-function dataset
Process sequence-function dataset file to have the following filename: SeqFxnDataset.pkl
Name the column with amino acid sequences 'sequence'
Name the column with functional score 'functional_score'

### Step 1: Train reward model
We train an ensemble of multi-layer perceptrons to predict the log fluoresence of CreiLOV variants in a DMS dataset. Our repository is setup to train on a sequence-function dataset file (SeqFxnDataset.pkl) with a sequence and functional_score column.

```python3 Training_Ensemble_of_reward_models.py > Training_Ensemble_of_reward_models.out```

Files generated:

### Step 2: Perform simulated annealing
Generate a small, high quality synthetic sequence dataset for SFT

```python3 script.py > script.out```

Files generated:

### Step 3: SFT
Supervise finetune pLM

```python3 script.py > script.out```

Files generated:

### Step 4: Perform PPO
Align SFT-pLM with proximal policy optimization

```python3 script.py > script.out```

Files generated:

### Step 5: Generate designs
Generate designs to characterize

```python3 script.py > script.out```

Files generated:

