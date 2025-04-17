import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from patho_bench.datasets.BaseDataset import BaseDataset
from patho_bench.experiments.utils.ClassificationMixin import ClassificationMixin
from patho_bench.experiments.BaseExperiment import BaseExperiment
import json

"""
Runs linear probing using the sklearn LogisticRegression model.
"""

class LinearProbeExperiment(ClassificationMixin, BaseExperiment):
    def __init__(self,
                 dataset: BaseDataset,
                 task_name: str,
                 num_classes: int,
                 num_bootstraps: int,
                 cost: float,
                 max_iter: int,
                 balanced_class_weights: bool,
                 results_dir: str,
                 **kwargs
                 ):
        '''
        Linear Probe experiment class.

        Args:
            dataset (BaseDataset): Dataset object
            task_name (str): Name of task (must match key in labels dict)
            num_classes (int): Number of classes for each task
            num_bootstraps (int): Number of bootstraps for confidence intervals
            cost (float): Regularization parameter(s) for Logistic Regression
            max_iter (int): Maximum number of iterations for logistic regression
            balanced_class_weights (bool): Whether to use balanced class weights
            results_dir (str): Path to save results,
            **kwargs: Additional arguments to save in config.json
        '''
        self.dataset = dataset
        self.task_name = task_name
        self.num_classes = num_classes
        self.num_bootstraps = num_bootstraps
        self.cost = cost
        self.max_iter = max_iter
        self.balanced_class_weights = balanced_class_weights
        self.results_dir = results_dir
        self.set_seed(seed=0)
        
        # Set kwargs as extra attributes for saving in config.json
        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(self):
        self.save_config(os.path.join(self.results_dir, 'config.json'))

        print(f'Running linprobe experiment with C = {self.cost}...')
        self.models = {}
        
        loop = tqdm(range(self.dataset.num_folds))
        for self.current_iter in loop:   # Loop through folds
            
            # Get dataloader for current fold
            train_dataloader = self.dataset.get_dataloader(self.current_iter, 'train')
            assert len(train_dataloader) == 1, f'Dataloader must return one batch with all samples, got {len(train_dataloader)}'
            all_train_samples = next(iter(train_dataloader))

            loop.set_description(f'Training on {len(all_train_samples["labels"][self.task_name])} samples')
            assert len(all_train_samples['slide']['features'].shape) == 2, f'Features must be 2-dimensional (num_samples x feature_dim), got shape: {all_train_samples["slide"]["features"].shape}'
            embedding_dim = all_train_samples['slide']['features'].shape[1]
            
            model = LogisticRegression(
                C = (embedding_dim * self.num_classes / 100) if self.cost == 'adaptive' else self.cost,
                max_iter=self.max_iter,
                n_jobs=-1,
                verbose=0,
                random_state=0,
                class_weight="balanced" if self.balanced_class_weights else None
            )
            model.fit(all_train_samples['slide']['features'], all_train_samples['labels'][self.task_name])
            self.models[self.current_iter] = model
            
        # After training all folds, attempt to validate if a val set is available
        self.validate()

    def test(self):
        '''
        Evaluate model on test set.
        '''
        self._eval(split='test')
    
    def validate(self):
        """
        Evaluate model on validation set.
        """
        self._eval(split='val')
    
    def _eval(self, split: str):
        """
        Shared evaluation logic for either 'val' or 'test' splits
        
        Args:
            split (str): 'val' or 'test'
        """
        all_labels_across_folds = []
        all_preds_across_folds = []
        all_scores_across_folds = []

        loop = tqdm(range(self.dataset.num_folds))
        for self.current_iter in loop:
            # Get dataloader for current fold
            eval_dataloader = self.dataset.get_dataloader(self.current_iter, split)
            if eval_dataloader is None:
                print(f'No {split} set found. Skipping...')
                return  # No data for this fold in the chosen split

            assert len(eval_dataloader) == 1, f'Dataloader must return one batch with all samples, got {len(eval_dataloader)}'
            all_eval_samples = next(iter(eval_dataloader))

            # Get labels and predictions
            loop.set_description(f'Running {split} split on {len(all_eval_samples["labels"][self.task_name])} samples')
            labels = all_eval_samples['labels'][self.task_name] # Shape: (num_samples,)
            preds = self.models[self.current_iter].predict_proba(all_eval_samples['slide']['features'])

            # Decide whether to report per-fold results (mean ± SD) or bootstrapped results (95% CI)
            if len(eval_dataloader.dataset) == 1 or self.dataset.num_folds == 1:
                # If only one fold or one sample per fold, will save results at end across all folds
                all_labels_across_folds.append(labels)
                all_preds_across_folds.append(preds)
            else:
                # If multiple folds and multiple samples per fold, save per-fold results
                per_fold_save_dir = os.path.join(self.results_dir, f'{split}_metrics', f'fold_{self.current_iter}')
                scores = self.classification_metrics(labels, preds, self.num_classes, saveto=os.path.join(per_fold_save_dir, 'metrics.json'))
                all_scores_across_folds.append(scores['overall'])

        # Summarize across folds or across bootstraps
        if len(all_labels_across_folds) > 0:
            # Perform bootstrapping and calculate 95% CI
            bootstraps = self.bootstrap(all_labels_across_folds, all_preds_across_folds, self.num_bootstraps)
            all_scores_across_folds = [
                self.classification_metrics(labels, preds, self.num_classes)['overall']
                for labels, preds in tqdm(bootstraps, desc=f'Computing {self.num_bootstraps} bootstraps')
            ]
            
            # Save bootstrapped metrics
            folder_path = os.path.join(self.results_dir, f"{split}_metrics")
            os.makedirs(folder_path, exist_ok=True)  
            for idx, metrics_dict in enumerate(all_scores_across_folds):
                folder_path_curr = os.path.join(folder_path, f"bootstrap_{idx}")
                os.makedirs(folder_path_curr, exist_ok=True)  

                file_path = os.path.join(folder_path_curr, "metrics.json")
                with open(file_path, "w") as f:
                    json.dump(metrics_dict, f, indent=4)

            summary = self.get_95_ci(all_scores_across_folds)
        else:
            # Report mean ± SD
            summary = self.get_mean_se(all_scores_across_folds)

        with open(os.path.join(self.results_dir, f'{split}_metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)