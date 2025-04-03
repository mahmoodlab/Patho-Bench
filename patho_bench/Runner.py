import os
import subprocess
import sys; sys.path.append('../')
import yaml
import json
import pandas as pd
from patho_bench.SplitFactory import SplitFactory
from patho_bench.ExperimentFactory import generate_arg_combinations, generate_exp_id, parse_task_code
import time
from tqdm import tqdm


class Runner:
    def __init__(self, args):
        """
        Store the command-line arguments or configuration in this Agent instance.

        You can either:
        - pass a parsed argparse 'args' object
        - or pass a dictionary with the same fields

        For simplicity, we'll assume 'args' is an argparse Namespace.
        """
        self.args = args

        # For convenience, keep local references to frequently-used args:
        self.conda_venv = args.conda_venv
        self.venv = args.venv
        self.saveto = args.saveto
        self.tasks_yaml = args.tasks_yaml
        self.experiment_type = args.experiment_type
        self.model_name = args.model_name
        self.model_kwargs_yaml = args.model_kwargs_yaml
        self.combine_slides_per_patient = args.combine_slides_per_patient
        self.hyperparams_yaml = args.hyperparams_yaml
        self.pooled_dirs_yaml = args.pooled_dirs_yaml
        self.splits_root = args.splits_root
        self.patch_dirs_yaml = args.patch_dirs_yaml
        self.gpu = args.gpu
        self.global_delay = args.global_delay
        self.delay_interval = args.delay_interval
        self.preserve = args.preserve

        # Convert self.model_name to list if it's a single string
        if isinstance(self.model_name, str):
            self.model_name = [self.model_name]

        self.session_name = args.tmux_id if args.tmux_id else os.path.basename(self.saveto)
        self.progress_file = os.path.join(self.saveto, "progress.txt")

    def run(self):
        """
        Core logic to create the tmux session, launch the tasks, and monitor progress.
        """
        # Check if session exists in tmux and optionally kill it
        self._maybe_kill_existing_session()
        self._create_tmux_session()
        
        # Build commands to run in each pane
        with open(self.tasks_yaml, 'r') as file:
            task_codes_list = yaml.safe_load(file)
        commands = self._build_commands(task_codes_list)

        # Save study config
        self._save_study_config(task_codes_list, commands)

        # Create a new window and set up the progress monitor
        self._setup_progress_monitor()

        # Create another window and send commands to split panes
        self._launch_tasks_in_panes(commands)

        # Finally, attach to the session
        self.run_tmux_command(f"tmux attach -t {self.session_name}")

    def _maybe_kill_existing_session(self):
        """
        Check if session exists in tmux and optionally kill it.
        """
        if self.run_tmux_command(f"tmux has-session -t {self.session_name}"):
            answer = input(
                f"\033[91mAre you sure you want to kill the existing session {self.session_name}? (y/n): \033[0m"
            )
            if answer.lower() == "y":
                self.run_tmux_command(f"tmux kill-session -t {self.session_name}")
                print(f"Existing session {self.session_name} killed.")
            else:
                print("Cancelled.")
                self.run_tmux_command(f"tmux attach -t {self.session_name}")
                sys.exit(0)

    def _create_tmux_session(self):
        """
        Create a new tmux session in detached mode.
        """
        self.run_tmux_command(f"tmux new-session -d -s {self.session_name}")

    def _build_commands(self, task_codes_list):
        """
        Build a list of shell commands to run in each parallel tmux pane.
        Each command may contain multiple sub-commands joined by "&&" if the tasks YAML contains space-separated task codes.
        For each model name, we create a set of commands for all task codes.
        """
        all_commands = []
        for model in self.model_name:
            model_commands = []
            for task_codes in task_codes_list:
                # For a pane that runs multiple tasks in series, we join them with "&&".
                command_pieces = []
                for task_code in task_codes.split(" "):
                    single_command = (
                        f"python ../patho_bench/scripts/sweep_single_task.py "
                        f"--experiment_type {self.experiment_type} "
                        f"--model_name {model} "
                        f"--model_kwargs_yaml {self.model_kwargs_yaml} "
                        f"--task_code {task_code} "
                        f"--combine_slides_per_patient {self.combine_slides_per_patient} "
                        f"--saveto {self.saveto} "
                        f"--hyperparams_yaml {self.hyperparams_yaml} "
                        f"--pooled_dirs_yaml {self.pooled_dirs_yaml} "
                        f"--patch_dirs_yaml {self.patch_dirs_yaml} "
                        f"--splits_root {self.splits_root} "
                        f"--gpu {self.gpu}"
                    )
                    command_pieces.append(single_command)
                model_commands.append(" && ".join(command_pieces))
            all_commands.extend(model_commands)
        return all_commands

    def _save_study_config(self, task_codes_list, commands):
        """
        Save the launch configuration (arguments, commands, etc.) to a YAML file for reproducibility.
        """
        os.makedirs(self.saveto, exist_ok=True)
        with open(self.hyperparams_yaml, 'r') as file:
            hyperparams_dict = yaml.safe_load(file)

        # Build a single dictionary with all info
        study_config = {
            **vars(self.args),          # all argparse fields
            "task_codes": task_codes_list,
            "hyperparams": hyperparams_dict,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "commands": commands,
        }
        save_path = os.path.join(self.saveto, "config.yaml")
        if os.path.exists(save_path):
            answer = input(f"\033[91mFile {save_path} already exists. Overwrite? (y/n): \033[0m")
            if answer.lower() == "y":
                with open(save_path, 'w') as f:
                    yaml.dump(study_config, f, sort_keys=False)
                print(f"Saved launch config to {save_path}.")
            else:
                print("Warning: Experiment will proceed without saving the launch config.")
        else:
            with open(save_path, 'w') as f:
                yaml.dump(study_config, f, sort_keys=False)
            print(f"Saved launch config to {save_path}.")

    def _setup_progress_monitor(self):
        """
        Create the progress window and run the progress monitor script.
        """
        self.run_tmux_command(f"tmux rename-window -t {self.session_name}:0 progress")
        self.run_tmux_command(f"tmux send-keys -t {self.session_name}:0 '{self._activate_venv()}' C-m")

        monitor_cmd = (
            f"python ../patho_bench/scripts/monitor_progress.py "
            f"--experiment_dir {self.saveto} "
            f"--splits_root {self.splits_root}"
        )
        self.run_tmux_command(f"tmux send-keys -t {self.session_name}:0 '{monitor_cmd}' C-m")

    def _activate_venv(self):
        """
        Get the command to activate the desired virtual environment.
        """
        if self.conda_venv:
            return f"conda activate {self.conda_venv}"
        elif self.venv:
            return f"source {self.venv}"
        else:
            raise ValueError("Please provide either a conda or non-conda virtual environment.")
    
    def _launch_tasks_in_panes(self, commands):
        """
        Create a new window, split it into panes, and send the appropriate commands to each pane.
        """
        self.run_tmux_command(f"tmux new-window -t {self.session_name} -n experiments")

        # Initialize the progress file
        with open(self.progress_file, "w") as f:
            for i, command in enumerate(commands):
                f.write(f"Pane {i} [INIT]: {command}\n")

        # Launch commands in each pane
        for i, cmd in enumerate(commands):
            if i > 0:
                self.run_tmux_command(f"tmux split-window -t {self.session_name}:1")
                self.run_tmux_command(f"tmux select-layout -t {self.session_name}:1 tiled")

            # Activate the environment in the pane
            self.run_tmux_command(
                f"tmux send-keys -t {self.session_name}:1.{i} "
                f"'{self._activate_venv()}' C-m"
            )

            # Add a delay to each pane to avoid overloading the system
            self.run_tmux_command(
                f"tmux send-keys -t {self.session_name}:1.{i} "
                f"'sleep {int(i*self.delay_interval) + self.global_delay * 60}' C-m"
            )

            # Update progress file to reflect running/success/error states
            pre = f"sed -i \"s/^Pane {i} .*/Pane {i} [RUNNING...]/\" {self.progress_file}"
            ok = f"sed -i \"s/^Pane {i} .*/Pane {i} [SUCCESS]/\" {self.progress_file}"
            err = f"sed -i \"s/^Pane {i} .*/Pane {i} [ERROR]/\" {self.progress_file}"
            suffix = "exit;" if not self.preserve else ""

            pane_command = (
                f"{pre}; "
                f"{cmd}; "
                f"if [ $? -eq 0 ]; then {ok}; echo -e \"\\033[92mDONE\\033[0m\"; {suffix} "
                f"else {err}; echo -e \"\\033[91mERROR\\033[0m\"; fi"
            )
            self.run_tmux_command(
                f"tmux send-keys -t {self.session_name}:1.{i} '{pane_command}' C-m"
            )
    
    @staticmethod
    def run_tmux_command(command):
        """Run a tmux command using subprocess."""
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0

    @staticmethod
    def collect_results(study_path, splits_root_dir):
        '''
        Collects the results from a study directory containing multiple experiments.
        The study directory must be in a specific format, as generated by ../tutorial/run.py.
        
        Parameters:
        - study_path (str): The path to the study directory.
        - splits_root_dir (str): The path to the splits root directory for loading metric. For retrieval experiments, 3 metrics will be collected: 'mAP@1, 'mAP@5', and 'mAP@10'.
        '''
        assert os.path.exists(study_path), f'Study directory {study_path} does not exist.'

        with open(os.path.join(study_path, 'config.yaml'), 'r') as f:
            exp_config = yaml.safe_load(f)
            model_names = exp_config['model_name']
            hyperparam_combos = generate_arg_combinations(exp_config['hyperparams'])
            
        all_results = []
        all_missing_results = []
        for task_code in tqdm(flatten_list(exp_config['task_codes']), desc=f'Collecting results from {study_path}'):
            train_source, test_source, task_name = parse_task_code(task_code) # Parse task_code into train_source, test_source, and task_name
            datasource_code = f'{train_source}=={test_source}' if test_source is not None else train_source
            
            # Get desired metric
            if exp_config['experiment_type'] == 'retrieval':
                metrics = ['mAP@1', 'mAP@5', 'mAP@10']
            else:
                metrics = SplitFactory.get_task_info(saveto = splits_root_dir, source = train_source, task = task_name)['metrics']
    
            for hyperparams in hyperparam_combos:
                for model_name in model_names:
                    for metric in metrics:
                        exp_dir = os.path.join(study_path,
                                                    datasource_code,
                                                    task_name,
                                                    f"{model_name}_{exp_config['experiment_type']}",
                                                    generate_exp_id(hyperparams))
                        
                        exp_results = {}
                        for split in ['val', 'test']:
                            exp_results[split] = {'formatted': None, 'mean': None, 'se': None, 'ci_low': None, 'ci_high': None}
                            results_path = os.path.join(exp_dir, f'{split}_metrics_summary.json')
                            if os.path.exists(results_path):
                                with open(results_path, 'r') as f:
                                    results_summary = json.load(f)
                                    
                                exp_results[split]['formatted'] = enforce_decimals(results_summary[metric]['formatted']) # Format the value and error to 3 decimal places
                                if ' ± ' in exp_results[split]['formatted']: # mean ± se
                                    exp_results[split]['mean'] = exp_results[split]['formatted'].split(' ± ')[0]
                                    exp_results[split]['se'] = exp_results[split]['formatted'].split(' ± ')[1]
                                elif ' (' in exp_results[split]['formatted']: # mean (95% CI)
                                    exp_results[split]['mean'] = exp_results[split]['formatted'].split(' (')[0]
                                    exp_results[split]['ci_low'] = exp_results[split]['formatted'].split(' (')[1].split('-')[0]
                                    exp_results[split]['ci_high'] = exp_results[split]['formatted'].split(' (')[1].split('-')[1].split(')')[0]
                            
                        all_results.append({
                            'source': datasource_code,
                            'task': task_name,
                            'model': f"{model_name}_{exp_config['experiment_type']}",
                            'metric': metric,
                            **hyperparams,
                            'val_formatted': exp_results['val']['formatted'],
                            'val_mean': exp_results['val']['mean'],
                            'val_se': exp_results['val']['se'],
                            'val_ci_low': exp_results['val']['ci_low'],
                            'val_ci_high': exp_results['val']['ci_high'],
                            'test_formatted': exp_results['test']['formatted'],
                            'test_mean': exp_results['test']['mean'],
                            'test_se': exp_results['test']['se'],
                            'test_ci_low': exp_results['test']['ci_low'],
                            'test_ci_high': exp_results['test']['ci_high']
                        })
                        if exp_results['test']['formatted'] is None:
                            all_missing_results.append(results_path)
                    
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(study_path, 'results_summary.csv'), index=False)
        if (len(df) - len(all_missing_results)) > 0:
            print(f'\033[92mCollected {len(df) - len(all_missing_results)} results. See {os.path.join(study_path, "results_summary.csv")}.\033[0m')
        
        if len(all_missing_results) > 0:
            with open(os.path.join(study_path, 'missing_results.txt'), 'w') as f:
                f.write('\n'.join(all_missing_results))
            print(f'\033[91mMissing {len(all_missing_results)} results. See {os.path.join(study_path, "missing_results.txt")} for details.\033[0m')
            
    @staticmethod
    def monitor_progress(experiment_dir):
        """
        Monitors the progress of an experiment by periodically reading and evaluating a 
        'progress.txt' file in the specified directory. The function waits in a loop, 
        checking whether any tasks are still running. If it detects changes in the tracked 
        progress, it prints real-time status updates including the count of completed 
        tasks, running tasks, and those with errors. The loop ends when no tasks are 
        marked as running, at which point a final completion message and the total 
        elapsed time are reported.

        Args:
            experiment_dir (str): Path to the experiment directory containing a 'progress.txt' file.
        """
        assert os.path.exists(experiment_dir), f'Experiment directory {experiment_dir} does not exist.'
        assert os.path.exists(os.path.join(experiment_dir, 'progress.txt')), f'Progress file not found in {experiment_dir}. Make sure you are using Runner.run() to launch the experiments.'
        start_time = time.time()
        progress = None
        while True:
            time.sleep(5)
            with open(os.path.join(experiment_dir, 'progress.txt'), 'r') as f:
                current_progress = f.read().splitlines()
            if not any(['RUNNING' in x for x in current_progress]):
                break # All tasks are done
            else:
                if progress != current_progress:
                    progress = current_progress
                    done = len([x for x in current_progress if 'SUCCESS' in x])
                    error = len([x for x in current_progress if 'ERROR' in x])
                    running = len([x for x in current_progress if 'RUNNING' in x])
                    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                    print(f'{elapsed_time} - Done: {done}, Running: {running}, Error: {error}')

        print(f"\033[94mExperiments finished in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}.\033[0m")
        
##############################################################################################################

def enforce_decimals(string):
    '''
    Enforces 3 decimal places for the value and error in the given string.
    Example: Given a string '0.5101845486886138 ± 0.04789558423491803', return '0.510 ± 0.048'.
    '''
    # Split the string into the value and the error
    if '±' not in string:
        return string
    value, error = str(string).split(' ± ')
    return f'{float(value):.3f} ± {float(error):.3f}'  # Round the value and error to 3 decimal places
    
def flatten_list(x):
    '''
    Given a list of space-separated strings, returns a list of strings.
    '''
    return [item for sublist in x for item in sublist.split(' ')]