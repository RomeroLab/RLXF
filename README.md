# RLXF  
Consolidated repository to perform RLXF.  

Perform the following steps to functionally align the protein language model (pLM) **ESM-2**.

## Key Terminology  
- **SFT**: Supervised Fine-tuning  
- **PPO**: Proximal Policy Optimization

## Step 0: Pre-process Sequence-Function Dataset  
Ensure the sequence-function dataset file is correctly formatted before proceeding.
- The dataset file should be named: **`SeqFxnDataset.pkl`**  
- The column containing amino acid sequences must be named: **`sequence`**  
- The column containing functional scores must be named: **`functional_score`**
- The column listing mutations must be named: **`mutations`** with the following format: G3E,L4N (no spaces, use 1-indexing i.e. first amino acid is M1 not M0 for start codon)

## Step 1: Train reward model
We train an ensemble of multi-layer perceptrons to predict the log fluoresence of CreiLOV variants in a DMS dataset. Our repository is setup to train on a sequence-function dataset file (SeqFxnDataset.pkl) with a sequence and functional_score column.

```python3 Training_Ensemble_of_reward_models.py > Training_Ensemble_of_reward_models.out```

Files generated:
- **SeqFxnDataset_splits.pkl**: datasplits for training, validation, and test sets
- **Loss_Curve.png**: plots average mse for ensemble vs. epoch
- **Test_Results.png**: plot of actual vs. predicted sequence function
- **Test_Results.csv**: contains 'MSE', 'Pearson R', and 'Spearman's Rho' metrics for test set
- also creates typical metrics files for each reward model in the log folder

## Step 2: Perform simulated annealing
Generate a small, high quality synthetic sequence dataset for SFT

```python3 simulated_annealing.py > simulated_annealing.out```

Files generated:
- **parameters.txt**: parameters used for simulated annealing
- **best_{num_mut}mut_v{i}.pickle**: contains best mutant found for trial
- **fitness_trajectory_{num_mut}mut_v{i}.csv**: contains scores for each step
- **traj_{num_mut}mut_v{i}.png**: plots scores vs. step for trial
- Optional: **close_sequences_{num_mut}mut_v{i}.pickle.pkl**: Use wt_functional_threshold to save sequences predicted to be have enhanced function relative to wildtype (parent sequence)

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

