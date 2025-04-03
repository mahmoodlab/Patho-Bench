import os
from tqdm import tqdm
from patho_bench.experiments.BaseExperiment import BaseExperiment
from patho_bench.datasets.BaseDataset import BaseDataset
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
import json
import numpy as np

"""
Runs Cox Proportional Hazards model with elastic net regularization on survival data.
"""

class CoxNetExperiment(BaseExperiment):
    def __init__(self,
                 dataset: BaseDataset,
                 task_name,
                 alpha,
                 l1_ratio,
                 max_iter,
                 num_bootstraps,
                 results_dir,
                 **kwargs
                 ):
        """
        Initializes the CoxNetExperiment with the given configuration.

        Args:
            dataset (BaseDataset): Dataset object
            task_name (str): Name of the task (must match key in labels dict).
            alpha (float): Regularization strength.
            l1_ratio (float): L1 ratio for CoxNet.
            max_iter (int): Maximum number of iterations for CoxNet.
            num_bootstraps (int): Number of bootstraps for confidence intervals.
            results_dir (str): Directory to save the results.
            **kwargs: Additional arguments to save in config.json
        """
        self.results_dir = results_dir
        self.dataset = dataset
        self.task_name = task_name
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.num_bootstraps = num_bootstraps
        self.set_seed(seed=0)
        
        # Set kwargs as extra attributes for saving in config.json
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def train(self):
        self.save_config(os.path.join(self.results_dir, 'config.json'))
        
        print(f'Running CoxNet experiment on task {self.task_name} with alpha={self.alpha}...')
        self.models = {}
        
        loop = tqdm(range(self.dataset.num_folds), desc='Training')
        for self.current_iter in loop:   # Loop through folds
            # Get dataloader for current fold
            train_dataloader = self.dataset.get_dataloader(self.current_iter, 'train')
            assert len(train_dataloader) == 1, f'Dataloader must return one batch with all samples, got {len(train_dataloader)}'
            all_train_samples = next(iter(train_dataloader))
            assert len(all_train_samples['slide']['features'].shape) == 2, f'Features must be 2-dimensional (num_samples x feature_dim), got shape: {all_train_samples["slide"]["features"].shape}'

            # Train model
            if self.l1_ratio > 0:
                loop.set_description(f'Training with elastic net regularization (L1 ratio = {self.l1_ratio}) on {len(all_train_samples["slide"]["features"])} samples...')
                model = CoxnetSurvivalAnalysis(alphas = [self.alpha], l1_ratio = self.l1_ratio, max_iter = self.max_iter)
            else:
                loop.set_description(f'Training with L2 regularization only on {len(all_train_samples["slide"]["features"])} samples...')
                model = CoxPHSurvivalAnalysis(alpha = self.alpha, n_iter=self.max_iter)

            train_y = self._to_structured_array(all_train_samples['labels']['extra_attrs'][f'{self.task_name}_event'], all_train_samples['labels']['extra_attrs'][f'{self.task_name}_days'])
            try:
                model.fit(all_train_samples['slide']['features'], train_y)
                self.models[self.current_iter] = model
            except Exception as e:
                print(f"\033[91mWARNING: {e}\033[0m")
                return
            
        # After training all folds, try validating
        self.validate()
                
    def test(self):
        '''
        Evaluate model on test set.
        '''
        self._eval(split='test')
        
    def validate(self):
        '''
        Evaluate model on validation set.
        '''
        self._eval(split='val')
        
    def _eval(self, split: str):
        """
        Shared evaluation logic for either 'val' or 'test' splits
        
        Args:
            split (str): 'val' or 'test'
        """
        all_scores_across_folds = []
        loop = tqdm(range(self.dataset.num_folds), desc=f'Evaluating on {split}')
        for self.current_iter in loop: # Loop through folds
            # Get dataloader for current fold
            eval_dataloader = self.dataset.get_dataloader(self.current_iter, split)
            if eval_dataloader is None:
                print(f'No {split} set found. Skipping...')
                return
            assert len(eval_dataloader) == 1, f'Dataloader must return one batch with all samples, got {len(eval_dataloader)}'
            all_eval_samples = next(iter(eval_dataloader))

            # Get labels and predictions
            loop.set_description(f'Running {split} split on {len(all_eval_samples["slide"]["features"])} samples...')
            eval_x = all_eval_samples['slide']['features']
            eval_y = self._to_structured_array(
                all_eval_samples['labels']['extra_attrs'][f'{self.task_name}_event'],
                all_eval_samples['labels']['extra_attrs'][f'{self.task_name}_days']
            )
            try:
                c_index = self.models[self.current_iter].score(eval_x, eval_y)
            except Exception as e:
                print(f"\033[91mWARNING: {e}\033[0m")
                return
                
            # Decide whether to report per-fold results (mean ± SD) or bootstrapped results (95% CI)
            if len(eval_x) == 1:
                # If only one sample per fold, save results at end across all folds
                raise NotImplementedError(f'Only one sample found in fold {self.current_iter} for {split} split, this is not yet supported.')
            elif self.dataset.num_folds == 1:
                # If only one fold and multiple samples per fold, perform bootstrapping and calculate 95% CI
                for _ in tqdm(range(self.num_bootstraps), desc=f'Computing {self.num_bootstraps} bootstraps'):
                    indices = np.random.choice(len(eval_x), len(eval_y), replace=True)
                    eval_x_resampled = eval_x[indices]
                    eval_y_resampled = eval_y[indices]
                    c_index = self.models[self.current_iter].score(eval_x_resampled, eval_y_resampled)
                    all_scores_across_folds.append({'cindex': c_index})
            else:
                # If multiple folds and multiple samples per fold, save per-fold results
                per_fold_save_dir = os.path.join(self.results_dir, f'{split}_metrics', f'fold_{self.current_iter}')
                os.makedirs(per_fold_save_dir, exist_ok=True)
                with open(os.path.join(per_fold_save_dir, 'metrics.json'), 'w') as f:
                    json.dump({'cindex': c_index}, f, indent=4)
                all_scores_across_folds.append({'cindex': c_index})

        if self.dataset.num_folds == 1:
            # Calculate 95% CI
            summary = self.get_95_ci(all_scores_across_folds)
            
            # Save all bootstrapped results
            with open(os.path.join(self.results_dir, 'all_bootstraps.json'), 'w') as f:
                json.dump(all_scores_across_folds, f, indent=4)
                
        elif len(all_scores_across_folds) > 0:
            # Report mean ± SD
            summary = self.get_mean_se(all_scores_across_folds)
        else:
            raise NotImplementedError(f'Bootstrapping across folds is not yet supported. Ensure that there are multiple test samples per fold.')

        with open(os.path.join(self.results_dir, f'{split}_metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
                
    @staticmethod
    def _to_structured_array(events, times):
        '''
        Converts events and times to a structured array.

        Args:
            events (np.array): Array of event indicators.
            times (np.array): Array of times.
        '''
        structured_array = np.empty(len(events), dtype=[('Event Indicator', 'bool'), ('Time', 'float')])
        structured_array['Event Indicator'] = events.numpy().astype(bool)
        structured_array['Time'] = times.numpy()
        return structured_array