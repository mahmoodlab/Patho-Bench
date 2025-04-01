import os
import yaml
import argparse
import sys; sys.path.append('../')
from patho_bench.SplitFactory import SplitFactory
from patho_bench.ExperimentFactory import ExperimentFactory, parse_task_code, make_list
from patho_bench.helpers.SpecialDtypes import SpecialDtypes

"""
##############################################################################################################
Run a hyperparameter sweep for a given experiment type, model, and task.
A dict of list of hyperparameters is specified in a config YAML. Combinations of hyperparameters are generated and the experiment is run for each combination in series.
Only one model and task code can be specified.

NOTE:
    It is recommended to run ../advanced_usage/run.py instead of this script.
    When you run ../advanced_usage/run.py, it will use tmux to run one or more tasks with flexible parallelism.
    In contrast, this script only supports a single task.

Usage:
    python sweep_single_task.py \
    --experiment_type linprobe \
    --model_name threads \
    --task_code bcnb--her2 \
    --combine_slides_per_patient True \
    --saveto ../../artifacts/experiments/single_task_example \
    --hyperparams_yaml "../../advanced_usage/configs/linprobe/linprobe.yaml" \
    --pooled_dirs_yaml "../../advanced_usage/configs/pooled_embeddings_paths.yaml" \
    --patch_dirs_yaml "../../advanced_usage/configs/patch_embeddings_paths.yaml" \
    --splits_root "../../artifacts/splits" \
    --gpu 0
##############################################################################################################
"""
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_type', type=str, help='Type of experiment to run')
    parser.add_argument('--model_name', type=str, help='Name of model to use')
    parser.add_argument('--model_kwargs_yaml', type=SpecialDtypes.none_or_str, default = None, help='Path to YAML file containing optional kwargs for initializing the model (e.g. an ABMIL model).')
    parser.add_argument('--task_code', type=str, help='Task code in format datasource--task_name or train_datasource==test_datasource--task_name')
    parser.add_argument('--combine_slides_per_patient', type=SpecialDtypes.bool, help='Whether to combine patches from multiple slides when pooling at case_id level. If False, will pool each slide independently. Note that some models e.g. GigaPath require this to be False.')
    parser.add_argument('--saveto', type=str, help='Directory to save the sweep')
    parser.add_argument('--hyperparams_yaml', type=str, help='Path to config YAML specifying hyperparameters to sweep over')
    parser.add_argument('--pooled_dirs_yaml', type=SpecialDtypes.none_or_str, default = None, help='Path to YAML file mapping data sources to pooled embeddings directories.')
    parser.add_argument('--patch_dirs_yaml', type=SpecialDtypes.none_or_str, default = None, help='Path to YAML file mapping data sources to patch embeddings directories.')
    parser.add_argument('--splits_root', type=SpecialDtypes.none_or_str, default = None, help='Root directory for downloading splits from HuggingFace')
    parser.add_argument('--path_to_split', type=SpecialDtypes.none_or_str, default = None, help='Local path to split file')
    parser.add_argument('--path_to_external_split', type=SpecialDtypes.none_or_str, default = None, help='Local path to external split file')
    parser.add_argument('--path_to_task_config', type=SpecialDtypes.none_or_str, default = None, help='Local path to task config file')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use for pooling. If -1, the best available GPU is used.')
    args = parser.parse_args()
    
    ###################################################################
    
    # Parse the task code
    train_source, test_source, task_name = parse_task_code(args.task_code)
    
    # Download the internal split from HuggingFace if not provided
    if args.path_to_split is None:
        path_to_split, path_to_task_config = SplitFactory.from_hf(args.splits_root, train_source, task_name)
    else:
        assert os.path.exists(args.path_to_split), f"Split file {args.path_to_split} does not exist."
        assert os.path.exists(args.path_to_task_config), f"Task config file {args.path_to_task_config} does not exist."
        path_to_split, path_to_task_config = args.path_to_split, args.path_to_task_config
    
    # Load the task config (used to get sample_col)
    task_info = SplitFactory.get_task_info(path_to_task_config)
    
    # Get hyperparameters to sweep over
    with open(args.hyperparams_yaml, 'r') as f:
        sweep_over = yaml.safe_load(f)
    
    # Get patch embeddings directories
    if args.patch_dirs_yaml is not None:
        with open(args.patch_dirs_yaml, 'r') as f:
            patch_dirs_dict = yaml.safe_load(f)
            patch_embeddings_dirs = make_list(patch_dirs_dict[train_source][args.model_name])
    else:
        patch_embeddings_dirs = None
        
    # Get pooled embeddings directories
    if args.pooled_dirs_yaml is not None:
        with open(args.pooled_dirs_yaml, 'r') as f:
            pooled_dirs_dict = yaml.safe_load(f)
            pooled_embeddings_dir = os.path.join(pooled_dirs_dict[train_source][args.model_name], f'by_{task_info["sample_col"]}') # Pooled embeddings are dependent on sample_col, for example, some tasks pool by case_id and some pool by slide_id
    else:
        pooled_embeddings_dir = None
        
    # Get external args
    # Download the external split from HuggingFace if not provided
    if test_source is None:
        path_to_external_split = None
        external_pooled_embeddings_dir = None
        external_saveto = None
    else:
        # Get path to external split
        if args.path_to_external_split is None:
            path_to_external_split, _ = SplitFactory.from_hf(args.splits_root, test_source, task_name)
        else:
            assert os.path.exists(args.path_to_external_split), f"External split file {args.path_to_external_split} does not exist."
            path_to_external_split = args.path_to_external_split
        
        # Update patch_embeddings_dirs to include external patch embeddings
        if patch_embeddings_dirs is not None:
            patch_embeddings_dirs += make_list(patch_dirs_dict[test_source][args.model_name])
        
        # Get external pooled embeddings directory
        if pooled_embeddings_dir is not None:
            external_pooled_embeddings_dir = os.path.join(pooled_dirs_dict[test_source][args.model_name], f'by_{task_info["sample_col"]}') # Pooled embeddings are dependent on sample_col, for example, some tasks pool by case_id and some pool by slide_id
        else:
            external_pooled_embeddings_dir = None
        
        # Get external saveto
        external_saveto = os.path.join(args.saveto, f'{train_source}=={test_source}', task_name)
        
    # Load model kwargs if provided
    if args.model_kwargs_yaml is not None:
        with open(args.model_kwargs_yaml, 'r') as f:
            model_kwargs = yaml.safe_load(f)
    else:
        model_kwargs = {}
        
    ###################################################################
    # Run the sweep
    ExperimentFactory.sweep(experiment_type = args.experiment_type,
                            split = path_to_split,
                            task_config = path_to_task_config,
                            saveto_root = os.path.join(args.saveto, train_source, task_name),
                            combine_slides_per_patient = args.combine_slides_per_patient,
                            sweep_over = sweep_over,
                            gpu = args.gpu,
                            pooled_embeddings_dir = pooled_embeddings_dir,
                            patch_embeddings_dirs = patch_embeddings_dirs,
                            model_name = args.model_name,
                            model_kwargs = model_kwargs,
                            external_split = path_to_external_split,
                            external_pooled_embeddings_dir = external_pooled_embeddings_dir,
                            external_saveto = external_saveto,
                            num_bootstraps = 100)