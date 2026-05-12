# Part 2 W&B Experiments

Run these on a GPU machine after logging into W&B:

```bash
wandb login
python part2_wandb/run_experiments.py --experiment all
```

For a single experiment:

```bash
python part2_wandb/run_experiments.py --experiment noam_vs_fixed
python part2_wandb/run_experiments.py --experiment scaling_ablation
python part2_wandb/run_experiments.py --experiment attention_heatmaps
python part2_wandb/run_experiments.py --experiment positional_ablation
python part2_wandb/run_experiments.py --experiment label_smoothing
```

For one individual run inside an experiment:

```bash
python part2_wandb/run_experiments.py --experiment noam_vs_fixed --run noam_scheduler
python part2_wandb/run_experiments.py --experiment noam_vs_fixed --run fixed_lr
python part2_wandb/run_experiments.py --experiment scaling_ablation --run attention_scaled
python part2_wandb/run_experiments.py --experiment scaling_ablation --run attention_unscaled
python part2_wandb/run_experiments.py --experiment positional_ablation --run sinusoidal_positional_encoding
python part2_wandb/run_experiments.py --experiment positional_ablation --run learned_positional_encoding
python part2_wandb/run_experiments.py --experiment label_smoothing --run label_smoothing_0_1
python part2_wandb/run_experiments.py --experiment label_smoothing --run label_smoothing_0_0
```

List all available names:

```bash
python part2_wandb/run_experiments.py --list
```

The runs log:

- train/validation loss curves
- learning rate curves
- prediction confidence
- query/key gradient norms for the attention scaling ablation
- final test BLEU
- encoder-head attention heatmaps for the attention visualization run

Use the logged runs to create the public W&B report required by the assignment.
