# RLXF
Consolidated repository to perform RLXF

Notes:
SFT: Supervised finetuning
PPO: Proximal policy optimization

### Step 1: Train reward model
We train an ensemble of multi-layer perceptrons to predict the log fluoresence of CreiLOV variants in a DMS dataset. Our repository is setup to train on a sequence_function_dataset.pkl file with a Sequence and Score column.

'''python3 script.py > script.out'''

Files generated:

### Step 2: Perform simulated annealing
Generate a small, high quality synthetic sequence dataset for SFT

'''python3 script.py > script.out'''

Files generated:

### Step 3: SFT
Supervise finetune pLM

'''python3 script.py > script.out'''

Files generated:

### Step 4: Perform PPO
Align SFT-pLM with proximal policy optimization

'''python3 script.py > script.out'''

Files generated:

### Step 5: Generate designs
Generate designs to characterize

'''python3 script.py > script.out'''

Files generated:

