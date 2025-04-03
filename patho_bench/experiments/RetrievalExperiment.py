import os
import json
import torch
from tqdm import tqdm
from patho_bench.datasets.BaseDataset import BaseDataset
from patho_bench.experiments.BaseExperiment import BaseExperiment
from patho_bench.experiments.utils.RetrievalMixin import RetrievalMixin
from patho_bench.experiments.utils.Retriever import Retriever


class RetrievalExperiment(RetrievalMixin, BaseExperiment):
    def __init__(self,
                 dataset: BaseDataset,
                 task_name,
                 num_classes,
                 num_bootstraps,
                 top_ks,
                 similarity,
                 use_centering,
                 results_dir,
                 **kwargs
                 ):
        """
        Initializes the RetrievalExperiment class.

        Args:
            dataset (BaseDataset): Dataset object
            task_name (str): Name of the task.
            num_classes (int): Number of classes.
            num_bootstraps (int): Number of bootstraps for confidence interval estimation.
            top_ks (list): List of top-k values for retrieval evaluation.
            similarity (str): Similarity metric to use ('l2', 'cosine', etc.).
            use_centering (bool): Whether to use centering in the similarity computation.
            results_dir (str): Directory to save results.
            **kwargs: Additional arguments to save in config.json.
        """
        self.dataset = dataset
        self.task_name = task_name
        self.num_classes = num_classes
        self.num_bootstraps = num_bootstraps
        self.top_ks = top_ks
        self.similarity = similarity
        self.use_centering = use_centering
        self.results_dir = results_dir
        self.set_seed(seed=0)
        
        # Set kwargs as extra attributes for saving in config.json
        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(self):
        self.save_config(os.path.join(self.results_dir, 'config.json'))

        print(f"Running retrieval experiment for task {self.task_name}...")
        self.models = {}

        for self.current_iter in tqdm(range(self.dataset.num_folds), desc="Training"):   # Loop through folds
            # Get dataloader for current fold
            train_dataloader = self.dataset.get_dataloader(self.current_iter, 'train')
            assert len(train_dataloader) == 1, f'Dataloader must return one batch with all samples, got {len(train_dataloader)}'
            all_train_samples = next(iter(train_dataloader))

            model = Retriever(self.similarity, self.use_centering)
            model.fit(all_train_samples['slide']['features'], all_train_samples['labels'][self.task_name])
            self.models[self.current_iter] = model
            
        # Now run validation if available
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

        loop = tqdm(range(self.dataset.num_folds), desc=f"Evaluating on {split}")
        for self.current_iter in loop:   # Loop through folds
            # Get dataloader for current fold
            eval_dataloader = self.dataset.get_dataloader(self.current_iter, split)
            if eval_dataloader is None:
                print(f'No {split} set found. Skipping...')
                return  # No data for this fold in the chosen split
            
            assert len(eval_dataloader) == 1, f'Dataloader must return one batch with all samples, got {len(eval_dataloader)}'
            all_eval_samples = next(iter(eval_dataloader))

            # Get labels and predictions
            labels = all_eval_samples['labels'][self.task_name]
            loop.set_description(f'Running {split} split on {len(labels)} samples')
            retrieved_labels = self.models[self.current_iter].retrieve(all_eval_samples['slide']['features'], max(self.top_ks)) # Shape: (num_samples, max(ks))

            # Decide whether to report per-fold results (mean ± SD) or bootstrapped results (95% CI)
            if len(eval_dataloader.dataset) == 1 or self.dataset.num_folds == 1:
                # If only one fold or one sample per fold, will save results at end across all folds
                all_labels_across_folds.append(labels)
                all_preds_across_folds.append(retrieved_labels)
            else:
                # If multiple folds and multiple samples per fold, save per-fold results
                per_fold_save_dir = os.path.join(self.results_dir, f'{split}_metrics', f'fold_{self.current_iter}')
                scores = self.retrieval_metrics(labels, retrieved_labels, self.top_ks, saveto=os.path.join(per_fold_save_dir, 'metrics.json'))
                all_scores_across_folds.append(scores['overall'])

        # Summarize results
        if len(all_labels_across_folds) > 0:
            # Perform bootstrapping and calculate 95% CI
            bootstraps = self.bootstrap(all_labels_across_folds, all_preds_across_folds, self.num_bootstraps)
            all_scores_across_folds = [
                self.retrieval_metrics(torch.from_numpy(labels), torch.from_numpy(preds), self.top_ks)['overall']
                for labels, preds in tqdm(bootstraps, desc=f'Computing {self.num_bootstraps} bootstraps')
            ]
            summary = self.get_95_ci(all_scores_across_folds)
            
            # Save all bootstrapped results
            with open(os.path.join(self.results_dir, f'all_bootstraps_{split}.json'), 'w') as f:
                json.dump(all_scores_across_folds, f, indent=4)
        else:
            # Report mean ± SD
            summary = self.get_mean_se(all_scores_across_folds)

        with open(os.path.join(self.results_dir, f'{split}_metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)