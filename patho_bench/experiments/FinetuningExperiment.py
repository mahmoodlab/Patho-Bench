import os
import numpy as np
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
import json
import warnings

# Import optimizers
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import AdamW

# Other imports
from patho_bench.datasets.BaseDataset import BaseDataset
from patho_bench.experiments.BaseExperiment import BaseExperiment
from patho_bench.experiments.utils.LoggingMixin import LoggingMixin
from patho_bench.experiments.utils.ClassificationMixin import ClassificationMixin
from patho_bench.experiments.utils.SurvivalMixin import SurvivalMixin


# Turn off tokenizer parallelism to avoid warnings from dataloader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
This file contains the FinetuningExperiment class, which is used to train and test supervised neural network models.
"""

class FinetuningExperiment(LoggingMixin, ClassificationMixin, SurvivalMixin, BaseExperiment):
    def __init__(self,
                 task_type: str,
                 dataset: BaseDataset,
                 batch_size: int,
                 model_constructor: callable,
                 model_kwargs: dict,
                 num_epochs: int,
                 accumulation_steps: int,
                 optimizer_config: dict,
                 scheduler_config: dict,
                 save_which_checkpoints: str,
                 num_bootstraps: int,
                 precision: torch.dtype,
                 device: str,
                 results_dir: str,
                 view_progress: str = 'bar',
                 lr_logging_interval: int = None,
                 seed: int = 7,
                 **kwargs):
        """
        Base class for all experiments.

        Args:
            task_type (str): Type of task. Can be 'classification' or 'survival'.
            dataset (BaseDataset): Dataset object
            batch_size (int): Batch size.
            model_constructor (callable): Model class which can be called to create model instance.
            model_kwargs: Arguments passed to model_constructor.
            num_epochs (int): Number of epochs.
            accumulation_steps (int): Number of batches to accumulate gradients over before stepping optimizer.
            optimizer_config: Optimizer config.
            scheduler_config: LR scheduler config.
            save_which_checkpoints (str): Mode of saving checkpoints.
            num_bootstraps (int): Number of bootstraps to use for computing 95% CI.
            precision (torch.dtype): Precision to use for training.
            device (str): Device to use for training.
            results_dir (str): Where to save results.
            view_progress (str, optional): How to log progress. Can be 'bar' or 'verbose'. Defaults to 'bar'.
            lr_logging_interval (int, optional): Interval at which to log learning rate to dashboard (in number of accumulation steps). Defaults to None (do not log).
            seed (int): Seed for reproducibility.
            **kwargs: Additional arguments to save in config.json
        """
        self.task_type = task_type
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_constructor = model_constructor
        self.model_kwargs = model_kwargs
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.save_which_checkpoints = save_which_checkpoints
        self.num_bootstraps = num_bootstraps
        self.precision = precision
        self.device = device
        self.results_dir = results_dir
        self.view_progress = view_progress
        self.lr_logging_interval = lr_logging_interval
        self.seed = seed
        self.set_seed(self.seed)
        
        # Set kwargs as extra attributes for saving in config.json
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Ensure that val set is nonempty if save_which_checkpoints is 'best-val-loss'
        if self.save_which_checkpoints == 'best-val-loss':
            assert self.dataset.get_subset(iteration = 0, fold = 'val') is not None, "Split must contain validation samples if save_which_checkpoints is 'best-val-loss'."

    def train(self):
        '''
        Runs training (and optionally validation) epochs for all folds of the experiment.
        '''
        print(f'\nExperiment dir: {self.results_dir}')
        self.save_config(os.path.join(self.results_dir, 'config.json'))
        self.train_results_dir = self.results_dir  # Store a copy of the training results dir for loading model in self.test(), in case want to save test results in a different directory

        ### Loop through folds
        for self.current_iter in range(self.dataset.num_folds):
            
            print("############################################################################################################")
            print(f"Training: Fold {self.current_iter + 1} of {self.dataset.num_folds}...")
            self.loggers = self.init_loggers(save_dir = os.path.join(self.results_dir, 'training_metrics', f'fold_{self.current_iter}'))

            ### Initialize train and val dataloaders
            self.dataloaders = {mode: self.dataset.get_dataloader(self.current_iter, mode, batch_size=self.batch_size) for mode in ['train', 'val']}
            
            ### Initialize model
            self.model = self.model_constructor(**self.model_kwargs, device = self.device)
            self.save_model_architecture(self.model, os.path.join(self.results_dir, f'model.txt'))
            
            ### Initialize optimizer and scheduler
            self.optimizer = self._init_optimizer()
            self.scheduler = self._init_scheduler()

            ### Prepare grad scaler
            # Only use GradScaler for FP16 training. bfloat16 does not require GradScaler: https://discuss.pytorch.org/t/bfloat16-training-explicit-cast-vs-autocast/202618/8
            try:
                self.grad_scaler = torch.amp.GradScaler('cuda', enabled = (self.precision == torch.float16)) 
            except:
                # Legacy (torch 2.0.0) implementation for compatibility with Gigapath
                self.grad_scaler = torch.cuda.amp.GradScaler(enabled = (self.precision == torch.float16))

            ### Initialize best loss and rank
            self.best_val_loss = 1e4            # Initialize to large number
            self.best_smooth_rank = 0           # Initialize to 0

            ### Prepare epoch loop
            if self.view_progress == 'bar':
                self.loop = tqdm(range(self.num_epochs))
            elif self.view_progress == 'verbose':
                self.loop = range(self.num_epochs)
            else:
                raise ValueError(f"view_progress must be 'bar' or 'verbose', got {self.view_progress} instead.")
            
            ### Loop through epochs
            for self.current_epoch in self.loop:
                for self.mode in ['train', 'val']:
                    if self.dataloaders[self.mode] is not None:
                        if self.view_progress == 'bar':
                            self.loop.set_description(f'      Epoch {self.current_epoch} {self.mode}')

                        start = time.time()
                        self._run_single_epoch()
                        end = time.time()

                        # Save progress to file
                        with open(os.path.join(self.results_dir, 'progress.txt'), 'a') as f:
                            f.write(f'DONE: Fold {self.current_iter + 1}/{self.dataset.num_folds} | {self.mode} | Epoch {self.current_epoch}\n')
                            
                        if self.view_progress == 'verbose':
                            print(f"Finished epoch = {self.current_epoch} in {end - start:.2f} seconds")
                            
        # After we finish all folds, try running a final "validation metrics" pass
        self.validate()

    def test(self):
        '''
        Evaluate the model on the test set for each fold.
        '''
        self._eval(split='test')

    def validate(self):
        """
        Evaluate the model on the validation set for each fold.
        """
        self._eval(split='val')
        
    def _eval(self, split: str):
        """
        Shared evaluation logic for either 'val' or 'test'. Similar to 
        your existing 'test()' method, but parameterized by `split`.
        """
        all_labels_across_folds = []
        all_preds_across_folds = []
        all_scores_across_folds = []

        ### Loop through folds
        loop = tqdm(range(self.dataset.num_folds))
        for self.current_iter in loop:
            ### Load the dataloader for this fold
            eval_dataloader = self.dataset.get_dataloader(self.current_iter, split, batch_size=1)
            if eval_dataloader is None:
                return
            loop.set_description(f'Running {split} split on {len(eval_dataloader.dataset)} samples')

            ### Get latest saved checkpoint for this fold
            checkpoint_dir = os.path.join(self.results_dir, 'checkpoints', f'fold_{self.current_iter}')
            ckpt_path = self._pick_checkpoint(checkpoint_dir)

            ### Load the model and freeze it
            model = self.model_constructor(**self.model_kwargs, device=self.device)
            model = self.load_checkpoint(model, ckpt_path)
            model = self.freeze(model)

            ### Gather labels and predictions
            labels, preds = self._accumulate_preds(eval_dataloader, model)

            ### Decide whether to report per-fold results (mean ± SD) or bootstrapped results (95% CI)
            if len(eval_dataloader.dataset) == 1 or self.dataset.num_folds == 1:
                # If only one fold or one sample per fold, will save results at end across all folds
                all_labels_across_folds.append(labels)
                all_preds_across_folds.append(preds)
            else:
                # If multiple folds and multiple samples per fold, save per-fold results
                per_fold_save_dir = os.path.join(self.results_dir, f'{split}_metrics', f'fold_{self.current_iter}')
                scores = self._compute_metrics(labels, preds, per_fold_save_dir)
                all_scores_across_folds.append(scores)

        # After collecting all folds, either do bootstrapping or an average across folds
        summary = self._finalize_metrics(split, all_labels_across_folds, all_preds_across_folds, all_scores_across_folds)

        with open(os.path.join(self.results_dir, f'{split}_metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
    def _accumulate_preds(self, dataloader, model):
        """
        Forward pass across the entire dataloader to collect predictions 
        and ground‐truth labels. 
        For classification: 
            returns (list_of_int, list_of_probs).
        For survival:
            returns (dict_of_arrays_for_events_times, array_of_risks).
        """
        labels_all = []
        preds_all = []

        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=self.precision, enabled=self.precision != torch.float32):
            for batch in dataloader:
                if self.task_type == 'classification':
                    label = batch['labels'][self.model_kwargs['task_name']].cpu().int().numpy().tolist()[0]
                    logits = model(batch, output='logits') # Forward pass
                    prob = torch.softmax(logits[0], dim=0).cpu().detach().numpy()
                    labels_all.append(label)
                    preds_all.append(prob)
                elif self.task_type == 'survival':
                    label = {'survival_event': batch['labels']['extra_attrs'][f'{self.model_kwargs["task_name"]}_event'],
                            'survival_time': batch['labels']['extra_attrs'][f'{self.model_kwargs["task_name"]}_days']}
                    logits = model(batch, output='logits') # Forward pass
                    risk = self._calculate_risk(logits[0]).cpu().detach().numpy()
                    labels_all.append(label)
                    preds_all.append(risk)
        
        # Convert to numpy arrays
        if self.task_type == 'classification':
            labels_all = np.array(labels_all)
            preds_all = np.array(preds_all)
        elif self.task_type == 'survival':
            labels_all = {k: np.array([v[k][0] for v in labels_all]) for k in labels_all[0].keys()}
            preds_all = np.array(preds_all)

        return labels_all, preds_all
    
    def _compute_metrics(self, labels, preds, save_dir):
        """
        Save metrics to file and return a dictionary of metrics.
        
        Args:
            labels (np.array or dict): Ground truth labels
            preds (np.array): Predictions
            save_dir (str): Directory to save metrics to
        """
        if self.task_type == 'classification':
            self.auc_roc(labels, preds, self.model_kwargs['num_classes'], saveto = os.path.join(save_dir, "roc_curves.png"))
            self.confusion_matrix(labels, preds, self.model_kwargs['num_classes'], saveto = os.path.join(save_dir, "confusion_matrices.png"))
            self.precision_recall(labels, preds, self.model_kwargs['num_classes'], saveto = os.path.join(save_dir, "pr_curves.png"))
            scores = self.classification_metrics(labels, preds, self.model_kwargs['num_classes'], saveto = os.path.join(save_dir, "metrics.json"))
            return scores['overall']
        elif self.task_type == 'survival':
            scores = self.survival_metrics(labels['survival_event'], labels['survival_time'], preds, saveto = os.path.join(save_dir, "metrics.json"))
            return scores

    def _finalize_metrics(self, split, labels_across_folds, preds_across_folds, scores_across_folds):
        """
        Combine per-fold results or do bootstrapping if single fold
        
        Arguments:
            split (str): Split name ('val' or 'test')
            labels_across_folds (list): List of labels across folds
            preds_across_folds (list): List of predictions across folds
            scores_across_folds (list): List of scores across folds
            
        Returns:    
            summary (dict): Dictionary of summary metrics
        """
        if len(labels_across_folds) > 0:
            # Perform bootstrapping and calculate 95% CI
            bootstraps = self.bootstrap(labels_across_folds, preds_across_folds, self.num_bootstraps)
            if self.task_type == 'classification':
                scores_across_folds = [self.classification_metrics(labels, preds, self.model_kwargs['num_classes'])['overall'] for labels, preds in tqdm(bootstraps, desc=f'Computing {self.num_bootstraps} bootstraps')]
            elif self.task_type == 'survival':
                scores_across_folds = [self.survival_metrics(labels['survival_event'], labels['survival_time'], preds) for labels, preds in tqdm(bootstraps, desc=f'Computing {self.num_bootstraps} bootstraps')]
            
            # Save bootstraps
            folder_path = os.path.join(self.results_dir, f"{split}_metrics")
            os.makedirs(folder_path, exist_ok=True)  
            for idx, metrics_dict in enumerate(scores_across_folds):
                folder_path_curr = os.path.join(folder_path, f"bootstrap_{idx}")
                os.makedirs(folder_path_curr, exist_ok=True)  

                file_path = os.path.join(folder_path_curr, "metrics.json")
                with open(file_path, "w") as f:
                    json.dump(metrics_dict, f, indent=4)

            return self.get_95_ci(scores_across_folds)
        else:
            # Report mean ± SE across folds
            return self.get_mean_se(scores_across_folds)

    def _pick_checkpoint(self, checkpoint_dir):
        """
        Picks the latest checkpoint from a directory. Returns None if no checkpoints are found.
        """
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
        available_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        
        if not available_checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
        if len(available_checkpoints) > 1:
            latest = max(available_checkpoints, key=lambda x: int(x.replace('.pt','').split('_')[-1]))
            warnings.warn(f"{len(available_checkpoints)} checkpoints found in {checkpoint_dir}. Using the latest checkpoint {latest}.")
            return os.path.join(checkpoint_dir, latest)
        else:
            return os.path.join(checkpoint_dir, available_checkpoints[0])
    
    def _run_single_epoch(self):
        """
        Runs a single training or validation epoch. After the epoch, the epoch metrics are stored in self.current_epoch_metrics.
        """
        
        # Set models to appropriate mode
        if self.mode == 'train':
            self.model.train()
            context_manager = torch.enable_grad()
        elif self.mode in ['val']:
            self.model.eval()
            context_manager = torch.inference_mode()
        else:
            raise ValueError('mode must be either "train", "val", or "test".')

        # Initialize performance trackers
        all_losses = []
        all_info = [] # This is additional info returned by the model along with the loss (e.g. predictions, targets, etc.)
        new_best_loss = False
        new_best_smooth_rank = False
        num_samples_processed = 0
        num_gradient_steps = 0

        # Loop over each batch in loader
        with context_manager:
            for batch_idx, batch in enumerate(self.dataloaders[self.mode]):
                num_samples_processed += len(batch['ids'])
                with torch.autocast(device_type='cuda', dtype=self.precision, enabled=self.precision != torch.float32):
                    loss, info = self.model(batch, output='loss')
                    loss = loss / self.accumulation_steps
                    assert isinstance(loss, torch.Tensor), f"Loss must be a tensor, got {loss} instead"
                    assert isinstance(info, list), f"Info must be a list on CPU, got {info} instead"

                # Update trackers
                all_losses.append(loss.cpu().detach().numpy())
                all_info.extend(info)

                # Backward pass if training (note that this is done outside autocast context manager)
                if self.mode == 'train':
                    self.grad_scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        self.grad_scaler.step(self.optimizer)
                        current_scale = self.grad_scaler.get_scale()
                        self.grad_scaler.update()
                        optimizer_skipped = (self.grad_scaler.get_scale() < current_scale) # If optimizer step was skipped due to gradient overflow, then scale will be reduced. In this case we must skip the scheduler step as well. See https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/8
                        self.optimizer.zero_grad()
                        num_gradient_steps += 1

                        # Log learning rate to dashboard on every lr_logging_interval accumulation steps
                        if self.scheduler_config and self.lr_logging_interval is not None and num_gradient_steps % self.lr_logging_interval == 0:
                            self.log_lr(batch_idx + self.current_epoch * len(self.dataloaders[self.mode]))                        

                        # Update scheduler on accumulation step if step_on is 'accumulation-step'
                        if self.scheduler_config and self.scheduler_config['step_on'] == 'accumulation-step' and not optimizer_skipped:
                            try:
                                # API for custom LR scheduler
                                partial_epoch_progress = (batch_idx + 1) / len(self.dataloaders[self.mode])
                                self.scheduler.step(total_progress = (self.current_epoch + partial_epoch_progress) / self.num_epochs)
                            except:
                                try:
                                    # Default API for built-in LR schedulers
                                    self.scheduler.step()
                                except:
                                    raise Exception(f"Error stepping scheduler on accumulation-step.")
                else:
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        num_gradient_steps += 1

                # Update progress bar
                if self.view_progress == 'bar' and (batch_idx + 1) % self.accumulation_steps == 0:
                    self.loop.set_postfix(num_batches = f'{batch_idx + 1}/{len(self.dataloaders[self.mode])}',
                                        num_samples = num_samples_processed,
                                        avg_loss = f'{(sum(all_losses)/num_gradient_steps):.4f}')

        # Update scheduler at end of epoch if step_on is 'epoch'
        if self.mode == 'train' and self.scheduler_config and self.scheduler_config['step_on'] == 'epoch' and not optimizer_skipped:
            self.scheduler.step()

        # Save current epoch metrics
        self.current_epoch_metrics = {"loss": all_losses, "info": all_info, 'per_sample_loss': sum(all_losses)/num_gradient_steps}
        self.compute_extra_metrics()

        # Update best smooth rank
        if isinstance(all_info[0], dict) and 'smooth_rank' in all_info[0].keys():
            smooth_rank = np.mean([info['smooth_rank'] for info in all_info])
            self.current_epoch_metrics['smooth_rank'] = smooth_rank # Add smooth rank to metrics
            if smooth_rank > self.best_smooth_rank:
                self.best_smooth_rank = smooth_rank
                new_best_smooth_rank = True
        else:
            assert self.save_which_checkpoints != 'best-smooth-rank', f"save_which_checkpoints cannot be 'best-smooth-rank' if smooth rank is not returned by the model."

        # Update best val loss
        if self.mode == 'val':
            avg_loss = np.mean(all_losses)
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                new_best_loss = True

        # Save checkpoints
        save_conditions = [self.save_which_checkpoints == 'all',
                           self.save_which_checkpoints == 'best-val-loss' and new_best_loss,
                           self.save_which_checkpoints == 'best-smooth-rank' and new_best_smooth_rank,
                           self.save_which_checkpoints.startswith('every-') and (self.current_epoch + 1) % int(self.save_which_checkpoints.split('-')[1]) == 0,
                           self.save_which_checkpoints.startswith('last-') and (self.current_epoch + 1) > self.num_epochs - int(self.save_which_checkpoints.split('-')[1])]
        if any(save_conditions):
            self.save_checkpoint(self.model, self.save_which_checkpoints, os.path.join(self.results_dir,
                                                                            'checkpoints', 
                                                                            f'fold_{self.current_iter}',
                                                                            f"epoch_{self.current_epoch}.pt"))

        self.log_loss(self.current_epoch) # Log loss to dashboard on epoch end
        self.log_smooth_rank(self.current_epoch) # Log smooth rank to dashboard on epoch end
    
    def _init_scheduler(self):
        '''
        Returns a scheduler. Supports one of the built-in schedulers or a custom scheduler class.
        '''
        if isinstance(self.scheduler_config['type'], str):
            # Using built-in scheduler
            if self.scheduler_config['type'] == 'plateau':
                return torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode=self.scheduler_config['mode'],
                    factor=self.scheduler_config['factor'],
                    patience=self.scheduler_config['patience'],
                    verbose=True)
            elif self.scheduler_config['type'] == 'step':
                return torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=self.scheduler_config['step_size'],
                    gamma=self.scheduler_config['gamma'])
            elif self.scheduler_config['type'] == 'cosine':
                assert self.accumulation_steps == 1, "CosineAnnealingLR scheduler is not compatible with gradient accumulation."
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.num_epochs if self.scheduler_config['step_on'] == 'epoch' else len(self.dataloaders['train']) * self.num_epochs,
                    eta_min=self.scheduler_config['eta_min'])
            elif self.scheduler_config['type'] == 'cosine_warm_restart':
                return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=self.scheduler_config['T_0'],
                    T_mult=self.scheduler_config['T_mult'],
                    eta_min=self.scheduler_config['eta_min'])
            else:
                raise NotImplementedError(f"Scheduler type {self.scheduler_config['type']} not implemented.")

        elif callable(self.scheduler_config['type']):
            # Using custom scheduler class
            try:
                default_scheduler_args = {
                    'base_lr': self.optimizer_config['base_lr'],
                    'max_epochs': self.num_epochs,
                    'accumulation_steps': self.accumulation_steps,
                    'len_dataloader': len(self.dataloaders['train']),
                }

                return self.scheduler_config['type'](
                    optimizer=self.optimizer,
                    default_scheduler_args=default_scheduler_args,
                    custom_scheduler_args=self.scheduler_config)
            except Exception as e:
                raise Exception(f"Error initializing custom scheduler: {e}. \nExpected init format: CustomScheduler(optimizer: Optimizer, default_scheduler_args: dict, custom_scheduler_args: dict)")

        else:
            raise ValueError(f"Scheduler type must be a string or a callable, got {self.scheduler_config['type']} instead.")

    def _init_optimizer(self):
        '''
        Initialize optimizer.
        '''
        optimizer_type = self.optimizer_config['type']
        extra_kwargs = {k: v for k, v in self.optimizer_config.items() if k not in ['type', 'get_param_groups', 'param_group_args', 'base_lr']}

        if 'get_param_groups' in self.optimizer_config:
            param_groups = self.optimizer_config['get_param_groups'](self.model, **self.optimizer_config['param_group_args'])
            assert isinstance(param_groups, list), "get_param_groups must return a list of dictionaries."
            assert len(param_groups) > 0, "get_param_groups must return a non-empty list of dictionaries."
        else:
            param_groups = self.model.parameters()

        if optimizer_type.lower() == "adam":
            return Adam(param_groups, self.optimizer_config['base_lr'], **extra_kwargs)
        elif optimizer_type.lower() == 'sgd':
            return SGD(param_groups, self.optimizer_config['base_lr'], **extra_kwargs)
        elif optimizer_type.lower() == "adamw":
            return AdamW(param_groups, self.optimizer_config['base_lr'], **extra_kwargs)
        else:
            raise NotImplementedError(f"Optimizer {optimizer_type} not implemented.")