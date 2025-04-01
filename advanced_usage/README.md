# 🛋️ Advanced Usage: Large-scale benchmarking

Patho-Bench offers a `Runner` class for large parallel runs with automatic GPU load balancing and experiment monitoring. Edit the following files:
1. `./advanced_usage/configs/tasks.yaml`: Define tasks. Tasks separated by spaces run in series, while newline-separated tasks run in parallel.
2. `./advanced_usage/configs/patch_embeddings_paths.yaml`: Dictionary of patch embedding locations, indexed by datasource and model name. Extract using Trident and provide paths here.
3. `./advanced_usage/configs/pooled_embeddings_paths.yaml`: Dictionary of pooled embedding locations, indexed by datasource and model name (follow same structure as `patch_embeddings_paths.yaml`). Extract using Patho-Bench (NOT Trident) and provide paths here.
4. `./advanced_usage/run.sh`: Define evaluation type and parameters.
5. `./advanced_usage/configs/`: Hyperparameter YAMLs for each evaluation framework. Provide multiple newline-separated values to run all hyperparameter combinations in series.

> [!NOTE]
> Be mindful of the hardware constraints of your machine when determining how many parallel experiments to run. Some evaluations, e.g. finetuning, are more compute-intensive than others. If you want to "set and forget" a large number of experiments but your machine is smol, you can set the `--delay_interval` argument in `run.sh` to a larger value. This will delay the start of each experiment by that many seconds, allowing you to run more experiments in parallel without overloading your machine.

1. Go to `./advanced_usage/configs/tasks.yaml` and input which tasks you want to run. Here's a potential set of tasks that you can run
    - **Note**: Space-separated task codes are run sequentially while newline-separated args are run in parallel.
    - **Note**: To run experiments in which you train on cohort A and test on cohort B, you need to construct the task code as follows in the `tasks.yaml` file: {train_dataset}=={test_dataset}--{task}. For example, `cptac_ccrcc==mut-het-rcc--BAP1_mutation` will run train BAP1 mutation prediction on CPTAC CCRCC and test on MUT-HET-RCC.
2. Depending on the evaluation framework, navigate to the correct folder at `./advanced_usage/configs/` and define the set of hyperparameters you want to run. For linear probe, you could define:
    ```yaml
    COST: # Regularization cost
    - 0.1
    - 0.5
    - 1.0
    - 10
    - adaptive

    BALANCED: # Balanced class weights
    - True
    ```
> [!NOTE]
> Instead of providing a list of `COST` values, you can set `COST: auto` to automatically sweep over `np.logspace(np.log10(10e-6), np.log10(10e5), num=45)`. This behavior can be modified in `ExperimentFactory.py`.

> [!NOTE]
> You can define model-specific hyperparameters in separate YAML files. For example, see `./advanced_usage/configs/fineune/*.yaml` for finetuning hyperparameters for several models evaluated in the [THREADS paper](https://arxiv.org/pdf/2501.16652). The `--hyperparams_yaml` argument in `run.sh` should point to the desired YAML file (see below).

3. Navigate to `./advanced_usage/run.sh` and edit the command to run the desired evaluation. See `./advanced_usage/run.py` for all possible arguments. For example:
    ```bash
    python run.py \
        --experiment_type linprobe \
        --model_name titan \                  # This can be a list of models, if desired. Just make sure your other arguments are model-agnostic.
        --tasks_yaml "configs/tasks.yaml" \
        --combine_slides_per_patient False \  # This parameter is different for different models. Titan requires this to be False.
        --saveto "../artifacts/example_runs/titan_linprobe" \
        --hyperparams_yaml "configs/linprobe/linprobe.yaml" \
        --pooled_dirs_yaml "configs/pooled_embeddings_paths.yaml" \
        --patch_dirs_yaml "configs/patch_embeddings_paths.yaml" \
        --splits_root "../artifacts/splits" \
        --conda_venv pathobench \
        --delay_interval 5 # Controls how much each pane is delayed by. Pane i will start after (i*delay_interval) seconds
    ```
4. Run `./run.sh`: This command will launch `tmux` windows for each parallel process and will close tmux windows automatically as the tasks are done.
    - You may need to make the script executable first: `chmod +x run.sh`
