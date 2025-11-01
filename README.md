# Reinforcement Learning from eXperimental Feedback (RLXF)
Reinforcement Learning from eXperimental Feedback (RLXF) is a framework for aligning the protein language model (pLM) **ESM-2** with experimentally derived notions of biomolecular function and can be adapted for your pLM of choice (Figure below). This enables the generative design of proteins with enhanced properties tailored to user-defined objectives. Our approach draws inspiration from the reinforcement learning techniques that aligned large language models with human preferences, resulting in transformative tools such as ChatGPT and Claude. In RLXF, a reward function, such as a supervised sequence-function predictor or any sequence scoring model, provides feedback to the pLM, guiding it to generate sequences with improved function. The outcome is a functionally aligned model that can be repeatedly sampled to produce diverse sequences optimized for the desired property.

We applied RLXF across five diverse protein classes to demonstrate its generalizability and effectiveness at generating optimized sequences by learning functional constraints beyond those captured during pre-training. As an in-depth case study, we aligned the 650M parameter ESM-2 model to experimental fluorescence data from the CreiLOV flavin-binding fluorescent protein. The aligned model learned to prioritize mutations that enhance fluorescence, many of which were missed by the base model. Experimental validation revealed the RLXF-aligned model generates a higher fraction of functional sequences, a greater number of sequences more fluorescent than CreiLOV, and the brightest oxygen-independent fluorescent protein variant reported to date. We provide data from these studies in the directory **`paper_analysis`**.

Check out our preprint here: https://www.biorxiv.org/content/10.1101/2025.05.02.651993v1.article-metrics

![Figure_1](https://github.com/user-attachments/assets/ebb952ce-137d-4af5-96da-d360a0d85370)

## Performing RLXF
Perform the following steps to functionally align the protein language model (pLM) **ESM-2**.

### Key Terminology  
- **SFT**: Supervised Fine-tuning  
- **PPO**: Proximal Policy Optimization

### Step 0: Pre-process Sequence-Function Dataset  
Ensure the sequence-function dataset file is correctly formatted before proceeding.
- The dataset file should be named: **`SeqFxnDataset.pkl`**  
- The column containing amino acid sequences must be named: **`sequence`**  
- The column containing functional scores must be named: **`functional_score`**
- The column listing mutations must be named: **`mutations`** with the following format: G3E,L4N (no spaces, use 1-indexing i.e. first amino acid is M1 not M0 for start codon)

### Step 1: Train reward model
We train an ensemble of multi-layer perceptrons to predict the log fluoresence of CreiLOV variants in a DMS dataset. Our repository is setup to train on a sequence-function dataset file (SeqFxnDataset.pkl) with a sequence and functional_score column.

```python3 Training_Ensemble_of_reward_models.py > Training_Ensemble_of_reward_models.out```

Files generated:
- **SeqFxnDataset_splits.pkl**: datasplits for training, validation, and test sets
- **logs/reward_model**: folder containing
  - metrics and hyperparameters for each reward model
- **reward_models**: folder containing
  - reward models as .ckpt files
  - **Loss_Curve.png**: average mse loss vs. epoch for ensemble of reward models
  - **Test_Results.png**: plot of actual vs. predicted sequence function
  - **Test_Results.csv**: contains 'MSE', 'Pearson R', and 'Spearman's Rho' metrics for test set

### Step 2: Generate synthetic sequence dataset for SFT via simulated annealing trials
We generate a small, high quality synthetic sequence dataset via simulated annealing trials

```python3 simulated_annealing.py > simulated_annealing.out```

Files generated:
- **unique_optimized_designs_from_simulated_annealing.pkl**: contains unique final optimized synthetic sequence for trials for SFT
- **all_optimized_designs_from_simulated_annealing.pkl**: contains final optimized synthetic sequence from each trial
- **simulated_annealing_results**: folder containing
  - **SA_mutation_distribution.png/svg**: contains heatmap of mutations in unique_optimized_designs_from_simulated_annealing.pkl sequences
  - **{num_muts}mut_{nsteps}steps**: folder containg
    - **parameters.txt**: parameters used for simulated annealing
    - **best_{num_mut}mut_v{i}.pickle**: contains best mutant found for trial
    - **fitness_trajectory_{num_mut}mut_v{version}.csv**: contains scores for each step
    - **traj_{num_mut}mut_v{i}.png**: plots scores vs. step for trial
    - Optional: **close_sequences_{num_mut}mut_v{version}.pickle**: Use wt_functional_threshold to save sequences predicted to be have enhanced function relative to wildtype (parent sequence)
    - **traj_{num_mut}mut_v{i}.png**: plots scores vs. step for trial
  
### Step 3: SFT
Supervise finetune pLM

```python3 running_SFT.py > running_SFT.out```

Files generated:
- **logs/SFT_{model_identifier}/version_{version}**: folder containing the following
  - **SFT_{model_identifier}.pt**: SFT pLM saved as .pt file
  - **{model_identifier}_fixed_mutated_designs_scores.csv**, **fixed_{model_identifier}_mutated_seqs.txt**, and **fixed_{model_identifier}_scores.npy**: sequence designs and scores from fixed model
  - **{model_identifier}_sft_mutated_designs_scores.csv**, **sft_{model_identifier}_mutated_seqs.txt**, and **sft_{model_identifier}_scores.npy**: sequence designs and scores from SFT model
  - **esm2_t33_650M_UR50D_metrics_vs_steps.png/svg**: various metrics vs. epoch monitored during SFT
  - metrics and hyperparameters for each reward model
  - single_mutant_probability_heatmaps: single mutant probabilities from pretrained or SFT pLM for wildtype sequenece or amino acid sequence with high confidence mutations

### Step 4: PPO
Align SFT-pLM with proximal policy optimization

```python3 running_PPO.py > running_PPO.out```

Files generated:
- **logs/PPO_{model_identifier}/version_{version}**: folder containing the following
  - **ema_aligned_{model_identifier}_v{version}_ep{epoch}.pt**: SFT pLM saved as .pt file, saved each epoch by default
  - **esm2_t33_650M_UR50D_design_scores_ep1.png/svg**: kdeplot of designs from pretrained, sft, and aligned models
  - **ema_aligned_{model_identifier}_mutated_designs_scores_ep1.csv**, **ema_aligned_{model_identifier}_mutated_seqs.txt**, and **ema_aligned_{model_identifier}_scores.npy**: sequence designs and scores from aligned model
  - **{model_identifier}_fixed_mutated_designs_scores.csv**, **fixed_{model_identifier}_mutated_seqs.txt**, and **fixed_{model_identifier}_scores.npy**: sequence designs and scores from fixed model
  - **{model_identifier}_sft_mutated_designs_scores.csv**, **sft_{model_identifier}_mutated_seqs.txt**, and **sft_{model_identifier}_scores.npy**: sequence designs and scores from SFT model
  - single_mutant_probability_heatmaps: single mutant probabilities from pretrained or SFT pLM for wildtype sequenece or amino acid sequence with high confidence mutations each epoch of training and during generation

## Training and reproducibility notes
- We trained the ensemble of reward models on one NVIDIA RTX A4500 GPU and performed simulated annealing on AMD EPYC 7302 16-Core Processor CPUs.
- We performed SFT and PPO with ESM-2 models on 1 NVIDIA L40S GPU.
- Packages for our conda enviroment:
  - *package*                 *version*
  - pytorch                   2.3.0
  - pytorch-cuda              12.1
  - pytorch-lightning         2.0.3
  - pytorch-mutex             1.0
  - torch-ema                 0.3
  - torchmetrics              1.1.2
  - torchtext                 0.18.0
  - numpy                     1.26.3
  - pandas                    1.5.3
  - transformers              4.40.1
  - matplotlib
  - scikit-learn
 
## 📬 Contact Me
If you have questions about this repository or encounter any issues, feel free to reach out:

Email: nlb51@duke.edu

GitHub Issues: Open an Issue

I welcome contributions, questions, and discussions related to RLXF.
 

