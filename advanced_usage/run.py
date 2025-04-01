import argparse
import sys; sys.path.append('../')
from patho_bench.helpers.SpecialDtypes import SpecialDtypes
from patho_bench.Runner import Runner

"""
Launches one or more tasks with flexible parallelism using tmux.
Flexible parallelism means that within each parallel instance, one or more tasks can be run in series.
You can specify which tasks to run and in what configuration using ./configs/tasks.yaml.
Hyperparameters for each task are specified in ./configs/{experiment_type}. All hyperparameter combinations are run in series for a given task.
"""

def main():
    parser = argparse.ArgumentParser(description="Launch multiple runs in parallel using tmux.")
    parser.add_argument("--experiment_type", type=str, choices=["linprobe", "coxnet", "retrieval", "finetune"], help="Type of experiment to run.")
    parser.add_argument("--model_name", type=str, nargs='+', help="Name(s) of the model(s) to use. Multiple models will run in parallel.")
    parser.add_argument("--tasks_yaml", type=str, default = "./configs/tasks.yaml", help="Path to the YAML file containing the task codes.")
    parser.add_argument('--combine_slides_per_patient', type=SpecialDtypes.bool, help='Whether to combine patches from multiple slides when pooling at case_id level. If False, will pool each slide independently.')
    parser.add_argument("--saveto", type=str, help="Save results to this directory. Basename of this dir will be used as the tmux session name, if --tmux_id is not provided.")
    parser.add_argument('--hyperparams_yaml', type=str, help='Path to config YAML specifying hyperparameters to sweep over')
    parser.add_argument('--pooled_dirs_yaml', type=SpecialDtypes.none_or_str, default = None, help='Path to YAML file mapping data sources to pooled embeddings directories.')
    parser.add_argument('--patch_dirs_yaml', type=SpecialDtypes.none_or_str, default = None, help='Path to YAML file mapping data sources to patch embeddings directories.')
    parser.add_argument('--splits_root', type=SpecialDtypes.none_or_str, default = None, help='Root directory for downloading splits from HuggingFace')
    parser.add_argument("--preserve", action="store_true", help="Preserve the finished tmux panes. If not provided, each tmux pane which successfully completes will be closed automatically.")
    parser.add_argument("--tmux_id", type=str, help="Optional tmux session name. If not provided, use the study_id.")
    parser.add_argument("--conda_venv", type=str, default='', help="Name of the conda venv to use within each tmux pane.")
    parser.add_argument("--venv", type=str, default='', help="Path to the non-conda virtual environment to use within each tmux pane. Format: /path/to/venv/bin/activate")
    parser.add_argument("--delay_factor", type=float, default=2, help="Factor by which to delay each pane. Pane i is delayed by i**delay_factor seconds.")
    parser.add_argument("--global_delay", type=int, default=0, help="Delay the whole study in minutes.")
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use for pooling. If -1, the best available GPU is used.')
    args = parser.parse_args()

    assert args.conda_venv or args.venv, "Please provide either a conda or non-conda virtual environment."

    agent = Runner(args)
    agent.run()


if __name__ == "__main__":
    main()
