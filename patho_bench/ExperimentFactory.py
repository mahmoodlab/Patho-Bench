import os
import numpy as np
import torch
from torch import nn
from patho_bench.experiments.LinearProbeExperiment import LinearProbeExperiment
from patho_bench.experiments.RetrievalExperiment import RetrievalExperiment
from patho_bench.experiments.CoxNetExperiment import CoxNetExperiment
from patho_bench.experiments.FinetuningExperiment import FinetuningExperiment
from patho_bench.experiments.GeneralizabilityExperimentWrapper import GeneralizabilityExperimentWrapper
from patho_bench.TrainableSlideEncoder import TrainableSlideEncoder
from patho_bench.SplitFactory import SplitFactory
from patho_bench.DatasetFactory import DatasetFactory
from patho_bench.helpers.GPUManager import GPUManager
from patho_bench.optim.NLLSurvLoss import NLLSurvLoss
from sklearn.utils.class_weight import compute_class_weight
from trident.slide_encoder_models.load import encoder_factory

"""
This file contains the ExperimentFactory class which is responsible for instantiating the appropriate experiment object.
"""

class ExperimentFactory:
                
    @staticmethod
    def linprobe(
                 split: str,
                 task_config: str,
                 pooled_embeddings_dir: str,
                 saveto: str,
                 combine_slides_per_patient: bool,
                 cost = 1,
                 balanced: bool = False,
                 gpu = -1,
                 external_split: str = None,
                 external_pooled_embeddings_dir: str = None,
                 external_saveto: str = None,
                 patch_embeddings_dirs: list[str] = None,
                 model_name: str = None,
                 num_bootstraps: int = 100): 
        '''
        Create linear probe experiment using slide-level embeddings.
        
        Args:
            split: str, path to local split file.
            task_config: str, path to task config file.
            pooled_embeddings_dir: str, path to folder containing pre-pooled embeddings (slide-level or patient-level). If empty, must provide patch_embeddings_dirs.
            saveto: str, path to save the results
            combine_slides_per_patient: bool, Whether to combine patches from multiple slides when pooling at case_id level. If False, will pool each slide independently.
            cost: list or float, cost for Linear Probe experiment
            balanced: bool, whether to use balanced class weights
            gpu: int, GPU id. If -1, the best available GPU is used.
            external_split: str, path to local split file for external testing.
            external_pooled_embeddings_dir: str, path to folder containing pooled embeddings for external testing. Only needed if external_split is not None.
            external_saveto: str, path to save the results of external testing. Only needed if external_split is not None.
            patch_embeddings_dirs: list of str, paths to folder(s) containing patch embeddings for given experiment. Only needed if pooled_embeddings_dir is empty.
            model_name: str, name of the model to use for pooling. Only needed if pooled_embeddings_dir is empty.
            num_bootstraps: int, number of bootstraps. Default is 100.
        '''
        split, task_info = SplitFactory.from_local(split, task_config)
        split.save(os.path.join(saveto, 'split.csv'), row_divisor = 'slide_id') # Save split to experiment folder for future reference
        
        # Load internal dataset
        internal_dataset = DatasetFactory.from_slide_embeddings(split = split,
                                                                task_name = task_info['task_col'],
                                                                pooled_embeddings_dir = pooled_embeddings_dir,
                                                                patch_embeddings_dirs = patch_embeddings_dirs,
                                                                combine_slides_per_patient = combine_slides_per_patient,
                                                                model_name = model_name,
                                                                gpu = gpu)

        # Initialize experiment
        experiment = LinearProbeExperiment(dataset = internal_dataset,
                                            combine_train_val = False,
                                            task_name = task_info['task_col'],
                                            num_classes = len(task_info['label_dict']),
                                            num_bootstraps = num_bootstraps,
                                            cost = cost,
                                            max_iter = 10000,
                                            balanced_class_weights = balanced,
                                            results_dir = saveto
                                            )

        if external_split is None:
            return experiment
        else:
            external_split, task_info = SplitFactory.from_local(external_split, task_config)
            external_split.remove_all_folds()
            external_split.assign_folds(num_folds=internal_dataset.num_folds, test_frac=1, val_frac=0, method='monte-carlo')  # Reassign all samples to test

            # Load external dataset        
            external_dataset = DatasetFactory.from_slide_embeddings(split = external_split,
                                                                    pooled_embeddings_dir = external_pooled_embeddings_dir,
                                                                    patch_embeddings_dirs = patch_embeddings_dirs,
                                                                    combine_slides_per_patient = combine_slides_per_patient,
                                                                    model_name = model_name,
                                                                    gpu = gpu)
            
            return GeneralizabilityExperimentWrapper(experiment,
                                                     external_dataset = external_dataset,
                                                    test_external_only = True,
                                                    saveto = external_saveto)
    
    @staticmethod
    def retrieval(
                 split: str,
                 task_config: str,
                 pooled_embeddings_dir: str,
                 saveto: str,
                 combine_slides_per_patient: bool,
                 similarity: str,
                 centering: bool,
                 gpu = -1,
                 external_split: str = None,
                 external_pooled_embeddings_dir: str = None,
                 external_saveto: str = None,
                 patch_embeddings_dirs: list[str] = None,
                 model_name: str = None,
                 num_bootstraps: int = 100): 
        '''
        Create retrieval experiment using slide-level embeddings.
        
        Args:
            model_name: str, name of the model
            train_source: str, name of the training data source
            test_source: str, name of the testing data source. If None, no generalizability experiment is run.
            task_name: str, name of the task
            patch_embeddings_dirs: list of str, paths to folder(s) containing patch embeddings for given experiment
            pooled_embeddings_root: str, path to folder containing pooled embeddings (slide-level or patient-level). If empty dir, embeddings will be pooled on the fly.
            saveto: str, path to save the results
            combine_slides_per_patient: bool, Whether to combine patches from multiple slides when pooling at case_id level. If False, will pool each slide independently
            similarity: str, similarity metric for Retrieval experiment
            centering: bool, whether to use centering in Retrieval experiment
            splits_root: str, path to root folder where splits are automatically saved from HuggingFace. Either splits_root or path_to_split must be provided.
            path_to_split: str, path to local split file. Either splits_root or path_to_split must be provided.
            path_to_external_split: str, path to local split file for external testing. If test_source is not None, either this or splits_root must be provided.
            path_to_task_config: str, path to task config file. If None, task config will be loaded from HF.
            num_bootstraps: int, number of bootstraps. Default is 100.
        '''
        split, task_info = SplitFactory.from_local(split, task_config)
        split.save(os.path.join(saveto, 'split.csv'), row_divisor = 'slide_id') # Save split to experiment folder for future reference
        
        # Load internal dataset
        internal_dataset = DatasetFactory.from_slide_embeddings(split = split,
                                                                task_name = task_info['task_col'],
                                                                pooled_embeddings_dir = pooled_embeddings_dir,
                                                                patch_embeddings_dirs = patch_embeddings_dirs,
                                                                combine_slides_per_patient = combine_slides_per_patient,
                                                                model_name = model_name,
                                                                gpu = gpu)

        # Initialize experiment
        experiment = RetrievalExperiment(dataset = internal_dataset,
                                           combine_train_val = False,
                                           task_name = task_info['task_col'],
                                           num_classes = len(task_info['label_dict']),
                                           num_bootstraps = num_bootstraps,
                                           top_ks = [1, 5, 10],
                                           similarity = similarity,
                                           use_centering = centering,
                                           results_dir = saveto
                                           )

        if external_split is None:
            return experiment
        else:
            external_split, task_info = SplitFactory.from_local(external_split, task_config)
            external_split.remove_all_folds()
            external_split.assign_folds(num_folds=internal_dataset.num_folds, test_frac=1, val_frac=0, method='monte-carlo')  # Reassign all samples to test

            # Load external dataset        
            external_dataset = DatasetFactory.from_slide_embeddings(split = external_split,
                                                                    pooled_embeddings_dir = external_pooled_embeddings_dir,
                                                                    patch_embeddings_dirs = patch_embeddings_dirs,
                                                                    combine_slides_per_patient = combine_slides_per_patient,
                                                                    model_name = model_name,
                                                                    gpu = gpu)
            
            return GeneralizabilityExperimentWrapper(experiment,
                                                     external_dataset = external_dataset,
                                                     test_external_only = True,
                                                     saveto = external_saveto)
    
    @staticmethod
    def coxnet(
            split: str,
            task_config: str,
            pooled_embeddings_dir: str,
            saveto: str,
            combine_slides_per_patient: bool,
            alpha: float,
            l1_ratio: float,
            gpu = -1,
            external_split: str = None,
            external_pooled_embeddings_dir: str = None,
            external_saveto: str = None,
            patch_embeddings_dirs: list[str] = None,
            model_name: str = None,
            num_bootstraps: int = 100): 
        '''
        Create CoxNet experiment using slide-level embeddings.
        
        Args:
            model_name: str, name of the model
            train_source: str, name of the training data source
            test_source: str, name of the testing data source. If None, no generalizability experiment is run.
            task_name: str, name of the task
            patch_embeddings_dirs: list of str, paths to folder(s) containing patch embeddings for given experiment
            pooled_embeddings_root: str, path to folder containing pooled embeddings (slide-level or patient-level). If empty dir, embeddings will be pooled on the fly.
            saveto: str, path to save the results
            combine_slides_per_patient: bool, Whether to combine patches from multiple slides when pooling at case_id level. If False, will pool each slide independently
            alpha: float, alpha parameter for CoxNet experiment
            l1_ratio: float, l1_ratio parameter for CoxNet experiment
            splits_root: str, path to root folder where splits are automatically saved from HuggingFace. Either splits_root or path_to_split must be provided.
            path_to_split: str, path to local split file. Either splits_root or path_to_split must be provided.
            path_to_external_split: str, path to local split file for external testing. If test_source is not None, either this or splits_root must be provided.
            path_to_task_config: str, path to task config file. If None, task config will be loaded from HF.
            num_bootstraps: int, number of bootstraps. Default is 100.
        '''
        split, task_info = SplitFactory.from_local(split, task_config)
        split.save(os.path.join(saveto, 'split.csv'), row_divisor = 'slide_id') # Save split to experiment folder for future reference
        
        # Load internal dataset
        internal_dataset = DatasetFactory.from_slide_embeddings(split = split,
                                                                task_name = task_info['task_col'],
                                                                pooled_embeddings_dir = pooled_embeddings_dir,
                                                                patch_embeddings_dirs = patch_embeddings_dirs,
                                                                combine_slides_per_patient = combine_slides_per_patient,
                                                                model_name = model_name,
                                                                gpu = gpu)

        # Initialize experiment
        experiment = CoxNetExperiment(dataset = internal_dataset,
                                            combine_train_val = False,
                                            task_name = task_info['task_col'],
                                            alpha = alpha,
                                            l1_ratio = l1_ratio,
                                            max_iter = 100000,
                                            num_bootstraps = num_bootstraps,
                                            results_dir = saveto
                                            )

        if external_split is None:
            return experiment
        else:
            external_split, task_info = SplitFactory.from_local(external_split, task_config)
            external_split.remove_all_folds()
            external_split.assign_folds(num_folds=internal_dataset.num_folds, test_frac=1, val_frac=0, method='monte-carlo')  # Reassign all samples to test

            # Load external dataset        
            external_dataset = DatasetFactory.from_slide_embeddings(split = external_split,
                                                                    pooled_embeddings_dir = external_pooled_embeddings_dir,
                                                                    patch_embeddings_dirs = patch_embeddings_dirs,
                                                                    combine_slides_per_patient = combine_slides_per_patient,
                                                                    model_name = model_name,
                                                                    gpu = gpu)
            
            return GeneralizabilityExperimentWrapper(experiment,
                                                     external_dataset = external_dataset,
                                                    test_external_only = True,
                                                    saveto = external_saveto)

    @staticmethod
    def finetune(model_name: str,
                 train_source: str,
                 task_name: str,
                 task_type: str,
                 patch_embeddings_dirs: list[str],
                 saveto: str,
                 combine_slides_per_patient: bool,
                 bag_size,
                 base_learning_rate,
                 gradient_accumulation,
                 weight_decay,
                 num_epochs,
                 scheduler_type: str,
                 optimizer_type: str,
                 balanced: bool,
                 save_which_checkpoints: str,
                 layer_decay = None,
                 gpu = -1,
                 test_source: str = None,
                 splits_root: str = None,
                 path_to_split: str = None,
                 path_to_external_split: str = None,
                 path_to_task_config: str = None,
                 num_bootstraps: int = 100):
        '''
        Create finetuning experiment, where the input is a bag of patch embeddings.

        Args:
            model_name: str, name of the model
            train_source: str, name of the training data source
            test_source: str, name of the testing data source. Currently, finetune only supports training and testing on the same data source (test_source must be None). Included for compatibility with ExperimentFactory.sweep().
            task_name: str, name of the task
            task_type: str, type of task. Can be 'survival' or 'classification'
            patch_embeddings_dirs: list of str, paths to folder(s) containing patch embeddings for given experiment
            saveto: str, path to save the results
            combine_slides_per_patient: bool, Whether to combine patches from multiple slides when pooling at case_id level. If False, will pool each slide independently
            bag_size: int or None, number of patches per bag
            base_learning_rate: float or None, base learning rate
            gradient_accumulation: int or None, gradient accumulation steps
            weight_decay: float or None, weight decay
            num_epochs: int or None, number of epochs
            scheduler_type: str, type of scheduler. Can be 'cosine' or 'gigapath'
            optimizer_type: str, type of optimizer. Can be 'AdamW' or 'gigapath'
            balanced: bool, whether to use balanced class weights
            save_which_checkpoints: str, which checkpoints to save
            layer_decay: float or None, layer decay for gigapath optimizer
            gpu: int, GPU id
            splits_root: str, path to root folder where splits are automatically saved from HuggingFace.
            path_to_split: str, path to local split file. Either splits_root or path_to_split must be provided.
            path_to_external_split: str, path to local split file for external testing. If test_source is not None, either this or splits_root must be provided. Currently, finetune only supports training and testing on the same data source (test_source must be None). Included for compatibility with ExperimentFactory.sweep().
            path_to_task_config: str, path to task config file. If None, task config will be loaded from HF.
            num_bootstraps: int, number of bootstraps. Default is 100.
        '''
        batch_size = 1

        assert task_type in ['survival', 'classification'], f'Invalid task type: {task_type}. Must be "survival" or "classification".'
        assert test_source is None, 'Finetuning only supports training and testing on the same data source for now. Please leave test_source as default (None).'
        
        ###### Get dataset ################################################################
        if path_to_split:
            assert path_to_task_config, 'path_to_task_config must be provided if path_to_split is provided.'
            split, task_info = SplitFactory.from_local(path_to_split, path_to_task_config, task_name)
        else:
            split, task_info = SplitFactory.from_hf(splits_root, train_source, task_name)
        split.save(os.path.join(saveto, 'split.csv'), row_divisor = 'slide_id') # Save split to experiment folder for future reference

        dataset = DatasetFactory.from_patch_embeddings(split = split,
                                                       task = task_name,
                                                       patch_embeddings_dirs = patch_embeddings_dirs,
                                                       combine_slides_per_patient = combine_slides_per_patient,
                                                       bag_size = bag_size)

        ###### Get loss ################################################################
        if task_type == 'survival':
            loss = NLLSurvLoss(alpha=0.0, eps=1e-7, reduction='mean')
        elif balanced:
            # Balanced loss is a dict of losses for each fold
            fold_weights = {fold: compute_class_weight('balanced', classes = np.array(sorted(split.unique_classes(task_name))), y = split.y(task_name, fold, 'train')) for fold in range(split.num_folds)}
            loss = {fold: nn.CrossEntropyLoss(weight = torch.from_numpy(weights).float()) for fold, weights in fold_weights.items()}
        else:
            loss = nn.CrossEntropyLoss()
        
        ###### Configure model ################################################################
        if model_name.startswith('abmil'):
            slide_encoder = encoder_factory(model_name,
                                            pretrained = False,
                                            freeze=False,
                                            input_feature_dim = 768,
                                            n_heads = 1,
                                            head_dim = 512,
                                            dropout = 0.25,
                                            gated = False)
        else:
            slide_encoder = encoder_factory(model_name, pretrained = False if 'randominit' in model_name else True, freeze=False)

        model_kwargs = {
                        'slide_encoder': slide_encoder,
                        'post_pooling_dim': slide_encoder.embedding_dim,
                        'task_name': task_name,
                        'num_classes': len(task_info['label_dict']),
                        'loss': loss
                        }

        ###### Configure scheduler ################################################################
        if scheduler_type == 'gigapath':
            from patho_bench.optim.GigaPathOptim import CustomLRScheduler
            scheduler_config = {'type': CustomLRScheduler,
                                'warmup_epochs': 1,
                                'min_lr': 0.000001,
                                'step_on': 'accumulation-step'}
        elif scheduler_type == 'cosine':
            scheduler_config = {'type': 'cosine',
                                'eta_min': 1e-8,
                                'step_on': 'accumulation-step'}
        else:
            raise NotImplementedError(f'Scheduler type {scheduler_type} not yet implemented. Please choose from "cosine" or "gigapath".')

        ###### Configure optimizer ################################################################
        if optimizer_type == 'gigapath':
            from patho_bench.optim.GigaPathOptim import param_groups_lrd
            optimizer_config = {'type': 'AdamW',
                                'base_lr': base_learning_rate * ((batch_size * gradient_accumulation) / 256),
                                'get_param_groups': param_groups_lrd,
                                'param_group_args': {'layer_decay': layer_decay,
                                                     'no_weight_decay_list': [],
                                                     'weight_decay': weight_decay},
                                }
        elif optimizer_type == 'AdamW':
            optimizer_config = {'type': 'AdamW',
                                'base_lr': base_learning_rate,
                                'weight_decay': weight_decay}
        else:
            raise NotImplementedError(f'Optimizer type {optimizer_type} not yet implemented. Please choose from "AdamW" or "gigapath".')
        
        ###### Configure experiment ################################################################
        experiment_kwargs = {
            'task_type': task_type,
            'dataset': dataset,
            'combine_train_val': False,
            'batch_size': batch_size,
            'model_constructor': TrainableSlideEncoder,
            'model_kwargs': model_kwargs,
            'num_epochs': num_epochs, # if nshots == 'all' else 500//(nshots * num_classes),
            'accumulation_steps': gradient_accumulation,
            'optimizer_config': optimizer_config,
            'scheduler_config': scheduler_config,
            'save_which_checkpoints': save_which_checkpoints,
            'num_bootstraps': num_bootstraps,
            'precision': slide_encoder.precision,
            'device': f'cuda:{gpu if gpu != -1 else GPUManager.get_best_gpu(min_mb=500)}',
            'results_dir': saveto
        }
            
        return FinetuningExperiment(**experiment_kwargs)
    
    @staticmethod
    def sweep(experiment_type: str,
              model_name: str,
              task_code: str,
              combine_slides_per_patient: bool,
              saveto: str,
              sweep_over: dict[list],
              splits_root: str,
              pooled_dirs_root: str = None,
              patch_dirs_dict: str = None,
              path_to_split: str = None,
              path_to_external_split: str = None,
              path_to_task_config: str = None,
              gpu = -1):
        '''
        Run a hyperparameter sweep for a given experiment configuration.
        
        Args:
            experiment_type (str): Type of experiment to run
            model_name (str): Name of model to use
            task_code (str): Task code in format datasource--task_name or train_datasource==test_datasource--task_name
            combine_slides_per_patient (bool): Whether to combine patches from multiple slides when pooling at case_id level. If False, will pool each slide independently
            saveto (str): Path to save the results
            sweep_over (dict[list]): Dictionary of hyperparameters to sweep over
            splits_root (str): Path to root folder where splits are automatically saved from HuggingFace.
            pooled_dirs_root (str): Path to root folder where pooled embeddings are saved. Subdirectories are automatically created for each datasource and pooling type. Not needed for finetuning.
            patch_dirs_dict (dict[str or list]): Dictionary of paths to patch embeddings, indexed by data source and patch encoder.
            gpu (int): GPU to use for pooling. If -1, the best available GPU is used.
        '''
        assert patch_dirs_dict is not None or patch_dirs_func is not None, 'One of patch_dirs_dict or patch_dirs_func must be provided.'
        train_source, test_source, task_name = parse_task_code(task_code) # Parse task_code into train_source, test_source, and task_name
        patch_embeddings_dirs = make_list(patch_dirs_dict[train_source][model_name])
        if test_source:
            patch_embeddings_dirs += make_list(patch_dirs_dict[test_source][model_name])
        
        args = {
            'train_source': train_source,
            'test_source': test_source,
            'task_name': task_name,
            'model_name': model_name,
            'combine_slides_per_patient': combine_slides_per_patient,
            'splits_root': splits_root,
            'path_to_split': path_to_split,
            'path_to_external_split': path_to_external_split,
            'path_to_task_config': path_to_task_config,
            'patch_embeddings_dirs': patch_embeddings_dirs,
            'gpu': gpu
        }
        
        # Iterate over all combinations of hyperparameters
        for hyperparams in generate_arg_combinations(sweep_over):
            args['saveto'] = os.path.join(saveto, train_source, task_name, f'{model_name}_{experiment_type}', generate_exp_id(hyperparams))
            
            if experiment_type == 'finetune':
                experiment = ExperimentFactory.finetune(**args, **hyperparams, task_type = 'survival' if task_name in ['OS', 'PFS', 'DSS'] else 'classification') # Infer task type from task name
            elif experiment_type == 'linprobe':
                experiment = ExperimentFactory.linprobe(**args, **hyperparams, pooled_embeddings_root = pooled_dirs_root)
            elif experiment_type == 'protonet':
                experiment = ExperimentFactory.protonet(**args, **hyperparams, pooled_embeddings_root = pooled_dirs_root)
            elif experiment_type == 'retrieval':
                experiment = ExperimentFactory.retrieval(**args, **hyperparams, pooled_embeddings_root = pooled_dirs_root)
            elif experiment_type == 'coxnet':
                experiment = ExperimentFactory.coxnet(**args,  **hyperparams, pooled_embeddings_root = pooled_dirs_root)
            else:
                raise NotImplementedError(f'Experiment type {experiment_type} not recognized. Please choose from "finetune", "linprobe", "protonet", "retrieval", or "coxnet".')
            
            experiment.train()
            experiment.test()

############################################################################################################
# Some helper functions
        
def parse_task_code(task_code):
    '''
    Parse task code into data source and task name.
    
    Args:
        task_code: str, in the format "data_source--task_name"
    
    Returns:
        str, str, str: train_source, test_source, task_name
    '''
    data_source, task_name = task_code.split('--')
    if '==' in data_source:
        train_source, test_source = data_source.split('==') # If running generalizability experiment, load split for internal dataset only
        assert train_source != test_source, f'train_source and test_source must be different when formatting task_code as "train_source==test_source--task_name". Did you mean to use {train_source}--{task_name} instead of {task_code}?'
        return train_source, test_source, task_name     
    else:
        train_source = data_source
        return train_source, None, task_name
    
def generate_exp_id(hyperparams):
    '''
    Generate a unique experiment ID from a dictionary of hyperparameters.
    
    Args:
        hyperparams: dict, hyperparameters
    
    Returns:
        str: experiment ID
    '''
    return '_'.join(sorted([f'{k}={v}' for k, v in hyperparams.items()]))
    
def generate_arg_combinations(variables):
    """
    Given a dict of lists, generate a list of dicts with all possible combinations of the input lists.
    Example: {"blr": [0.01, 0.1], "wd": [0.001, 0.01]} -> [{"blr": 0.01, "wd": 0.001}, {"blr": 0.01, "wd": 0.01}, {"blr": 0.1, "wd": 0.001}, {"blr": 0.1, "wd": 0.01}]
    
    Parameters:
    - variables (dict[list]): A dictionary where the keys are the variable names and the values are lists of values.
    
    Returns:
    - list[dict]: A list of dictionaries, each representing a combination of the input variables.
    """
    from itertools import product
    # If cost = 'auto', then automatically sweep over a range of costs (intended for linprobe)
    if 'auto' in make_list(variables.get('COST')):
        assert len(make_list(variables['COST'])) == 1, f'If setting cost to "auto", then only one cost value is allowed. Received {make_list(variables["COST"])}'
        variables['cost'] = list(np.logspace(np.log10(10e-6), np.log10(10e5), num=45))
        
    variables = {k.lower(): make_list(v) for k, v in variables.items()} # Ensure all values are lists and convert keys are lowercase
    return [dict(zip(variables.keys(), combination)) for combination in product(*variables.values())]
        
def make_list(x):
    '''
    Convert input to list if it is not already a list.
    '''
    return x if isinstance(x, list) else [x]