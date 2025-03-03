import os
import shutil
import yaml
import datasets
from patho_bench.datasets.DataSplit import DataSplit


class SplitFactory:    
    @staticmethod
    def from_local(path_to_split, path_to_config):
        '''
        Returns the datasplit from a local path.
        
        Args:
            path_to_split (str): Path to the split
            path_to_config (str): Path to the task config file
            
        Returns:
            split (DataSplit): Split object
            task_info (dict): Task metadata
        '''
        assert os.path.exists(path_to_split), f"Path to split {path_to_split} does not exist locally."
        assert os.path.exists(path_to_config), f"Path to split config {path_to_config} does not exist locally."
        
        task_info = SplitFactory.get_task_info(path_to_config=path_to_config)
        split = DataSplit(path = path_to_split,
                        id_col = task_info['sample_col'],
                        attr_cols = task_info['extra_cols'] + ['slide_id'],
                        label_cols = [task_info['task_col']])
        return split, task_info
    
    @staticmethod
    def from_hf(saveto, source, task):
        '''
        Downloads the split for a given source and task if it does not exist locally.
        
        Args:
            saveto (str): Path to save the split
            source (str): Name of source dataset
            task (str): Name of task
            
        Returns:
            path_to_split (str): Path to the split
            path_to_config (str): Path to the task config file
        '''
        path_to_split = os.path.join(saveto, source, task, f'k=all.tsv')
        path_to_config = os.path.join(saveto, source, task, 'config.yaml')
        if not os.path.exists(path_to_split):
            # Download split from HuggingFace
            try:
                print(f"Downloading split for task {task} from {source}...")
                datasets.load_dataset(
                    'MahmoodLab/Patho-Bench', 
                    cache_dir=saveto,
                    dataset_to_download=source,     # Throws error if source not found
                    task_in_dataset=task,           # Throws error if task not found in dataset
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Error: {e}")
                raise Exception(f"Could not load split for task {task}. Please ensure the split exists on HuggingFace.")
            
            # Cleanup HuggingFace cached files
            shutil.rmtree(os.path.join(saveto, 'MahmoodLab___patho-bench'))
            shutil.rmtree(os.path.join(saveto, '.cache'))
            for root, dirs, files in os.walk(saveto):
                for file in files:
                    if file.endswith('.lock'):
                        os.remove(os.path.join(root, file))
                        
        return path_to_split, path_to_config
    
    @staticmethod
    def get_task_info(path_to_config = None,
                      saveto = None,
                      source = None,
                      task = None):
        '''
        Returns the task metadata for a given source and task.
        
        Args:
            path_to_config (str): Path to the task config file. If None, will download the config file from HuggingFace.
            saveto (str): Path to save the split. Required if path_to_config is None.
            source (str): Name of source dataset. Required if path_to_config is None.
            task (str): Name of task. Required if path_to_config is None.
            
        Returns:
            task_info (dict): Task metadata
        '''
        if path_to_config is None:
            assert saveto is not None, "saveto must be provided if path_to_config is None."
            assert source is not None, "source must be provided if path_to_config is None."
            assert task is not None, "task must be provided if path_to_config is None."
            _, path_to_config = SplitFactory.from_hf(saveto, source, task)
            
        with open(path_to_config, 'r') as task_info:
            task_info = yaml.safe_load(task_info)
            
        # Check that task_info has the required format
        assert 'sample_col' in task_info and isinstance(task_info['sample_col'], str), f"sample_col (str) not found in task config at {path_to_config}."
        assert 'extra_cols' in task_info and isinstance(task_info['extra_cols'], list), f"extra_cols (list[str]) not found in task config at {path_to_config}."
        assert 'label_dict' in task_info and isinstance(task_info['label_dict'], dict), f"label_dict (dict) not found in task config at {path_to_config}."
        assert 'metrics' in task_info and isinstance(task_info['metrics'], list), f"metrics (list[str]) not found in task config at {path_to_config}."
            
        return task_info