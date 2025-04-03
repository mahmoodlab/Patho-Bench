import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as compute_split
from sklearn.model_selection import StratifiedKFold
from patho_bench.config.ConfigMixin import ConfigMixin
from IPython.display import display
from patho_bench.config.JSONSaver import JSONsaver

"""
This file contains the Split() class.
Used to create and/or load pathology slide datasets for training and evaluation.
"""

class DataSplit(ConfigMixin):
    def __init__(self,
                 path: str,
                 id_col: str,
                 attr_cols: list,
                 label_cols: list,
                 skip_labels: dict = None,
                 ignore_ids: list = None,
                 verbose: bool = True):
        '''
        Initializes the Split object by loading the dataset and setting up attributes.

        Args:
            path (str): Path to the split file.
            id_col (str): Column to use as ID.
            attr_cols (list): Attribute columns to include at the top level of the dict for each sample.
            label_cols (list): Columns to use for class labels.
            skip_labels (dict): If any label column has the corresponding value, the sample will be skipped.
            ignore_ids (list): List of IDs to ignore.
            verbose (bool): If True, will print information about the split when loaded.
        '''
        self.path = path
        self.id_col = id_col
        self.attr_cols = attr_cols
        self.label_cols = label_cols
        self.skip_labels = skip_labels
        self.ignore_ids = ignore_ids

        # Load split
        assert os.path.exists(self.path), f'{self.path} does not exist.'
        self.data = pd.read_csv(self.path, sep='\t' if self.path.endswith('.tsv') else ',', dtype={'case_id': str, 'slide_id': str, 'id': str})
        self.data = self.convert_to_json(self.data)  # Convert to a list of dicts for easier data access
        self.num_folds = len(self.data[0]['folds']) if 'folds' in self.data[0] else 0

        if verbose:
            print(f'Loaded split from {self.path} with {len(self.data)} samples and {self.num_folds} folds assigned.')

    def convert_to_json(self, df):
        '''
        Convert a splits dataframe to list of dicts for easier data access.
        This method is automatically run when loading a split from a .csv or .tsv file.

        Args:
            df (pd.DataFrame): DataFrame to convert
        '''
        # Detect if there is an index column that should be dropped
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
            
        # Drop any rows with invalid labels
        df = df.dropna(subset=self.label_cols)
        if self.skip_labels:
            for label_name, skip_vals in self.skip_labels.items():
                df = df[~df[label_name].isin(skip_vals)]
            
        samples_list = []
        for sample_id in df[self.id_col].unique():
            
            # Subset to rows with the current sample ID
            sample_df = df[df[self.id_col] == sample_id]
            sample_dict = {'id': str(sample_id)}

            if self.ignore_ids and str(sample_id) in [str(id) for id in self.ignore_ids]:
                print(f'Ignoring sample with ID {sample_id}.')
                continue
            
            # Get labels
            if self.label_cols:
                
                # Get all labels as dict
                labels = {col: sample_df[col].tolist() for col in self.label_cols}
                    
                # Check that all slides of sample have the same label combinations
                unique_label_sets = sample_df.drop_duplicates(subset=self.label_cols, keep='first')
                assert len(unique_label_sets) == 1, f'Inconsistent labels found for sample(s) with ID {sample_id}.\n{unique_label_sets}'
                
                sample_dict['labels'] = {k:v[0] for k, v in labels.items()} # Convert list to single value

            # Get folds from CSV, if available
            folds = []
            fold_columns = [col for col in sample_df.columns if 'fold_' in col]
            if len(fold_columns) > 0:
                fold_columns.sort(key=lambda x: int(x.split('_')[-1]))
                for col in fold_columns:
                    folds_for_sample = set(sample_df[col].values)
                    assert len(folds_for_sample) == 1, f'Inconsistent {col} found for sample ID {sample_id}.'
                    folds.append(folds_for_sample.pop())
                sample_dict['folds'] = folds

            # Add attr_cols to sample_dict
            if self.attr_cols:
                for attr_col in self.attr_cols:
                    sample_dict[attr_col] = sample_df[attr_col].tolist()

            samples_list.append(sample_dict)
        
        return samples_list
    
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        info = f"Split with {len(self.data)} samples and {self.num_folds} folds assigned.\n"
        info += "First 5 samples:\n"
        for i in range(min(5, len(self.data))):
            info += f"{self.data[i]}\n"
        return info

    def __add__(self, other):
        """
        Enables the addition of two Split instances to concatenate their data.

        Args:
            other (Split): Another Split instance to concatenate with self.

        Returns:
            Split: A new Split instance with combined data from self and other.
        """
        if not isinstance(other, DataSplit):
            raise TypeError(f"Cannot add Split with type {type(other)}")

        # Check if the configurations are compatible
        if self.label_cols != other.label_cols:
            raise ValueError("Cannot add Splits with different label columns.")
        if self.num_folds != other.num_folds:
            raise ValueError("Cannot add Splits with different numbers of folds.")
        
        # Ensure no id conflicts
        self_ids = set([sample['id'] for sample in self.data])
        other_ids = set([sample['id'] for sample in other.data])
        if len(self_ids.intersection(other_ids)) > 0:
            raise ValueError(f"Cannot add Splits with overlapping IDs.")

        # Combine the data
        combined_data = self.data + other.data

        # Create a new Split instance with the combined data
        # Here, we'll use the configuration of the first Split
        new_split = DataSplit(path=self.path, id_col=self.id_col, attr_cols=self.attr_cols, label_cols=self.label_cols, skip_labels=self.skip_labels, ignore_ids=self.ignore_ids)
        new_split.data = combined_data
        new_split.num_folds = self.num_folds

        return new_split
    
    def convert_labels_to_integers(self, label_attr, int_to_label = None):
        '''
        Converts all labels to integers and prints a dict of label values.

        Args:
            label_attr (str): Name of the label attribute to convert to integers
            int_to_label (dict): Dictionary mapping integers to labels. If not provided, will generate mapping from sorted unique labels.
        '''
        if int_to_label is None:
            # For each label, get all unique values
            unique_labels = set([sample['labels'][label_attr] for sample in self.data])

            # Make dict of label values
            label_to_int = {label: i for i, label in enumerate(sorted(unique_labels))}
            print('Generated int-to-label mapping:')
            display({str(v): k for k, v in label_to_int.items()})
        else:
            print('Using provided int-to-label mapping:')
            display(int_to_label)
            label_to_int = {v: int(k) for k, v in int_to_label.items()}

        # Convert labels to integers
        for sample in self.data:
            assert sample['labels'][label_attr] in label_to_int, f'Label {sample["labels"][label_attr]} (dtype {type(sample["labels"][label_attr])}) not found in label-to-int mapping.'
            sample['labels'][label_attr] = label_to_int[sample['labels'][label_attr]]

    @staticmethod
    def is_integer(value):
        try:
            # Try converting the value to an integer
            int_value = int(value)
            # Check if the converted value is still equal to the original
            if int_value == float(value):
                return True
        except (ValueError, TypeError):
            return False
        return False

    def save(self, export_to, row_divisor = None):
        '''
        Save the split as a CSV, TSV, or JSON.

        Args:
            export_to (str): Path to save the exported dataframe
            row_divisor (str): For CSV or TSV, the name of toplevel attribute to split rows by. If None, will export one row per sample.
        '''
        os.makedirs(os.path.dirname(export_to), exist_ok=True)
        if export_to.endswith('.json'):
            with open(export_to, 'w') as f:
                json.dump(self.data, f, indent=4, cls = JSONsaver)
            return
        
        # Make dataframe for CSV or TSV export
        rows = []
        for sample in self.data:
            # Initialize row for sample with all attributes except labels and folds
            row = {k: v for k, v in sample.items() if k not in ['id', 'labels', 'folds']}

            # Add label columns
            for label_col, label in sample['labels'].items():
                row[label_col] = label

            # Add fold columns
            for fold, fold_assignment in enumerate(sample['folds']):
                row[f'fold_{fold}'] = fold_assignment

            # Change lists of length 1 to single values
            for k, v in row.items():
                if isinstance(v, list) and len(set(v)) == 1:
                    row[k] = v[0]

            if row_divisor is not None:
                # Make a new copy of the row for each unique value of the row_divisor
                for unique_val in sample[row_divisor]:
                    new_row = row.copy()
                    new_row[row_divisor] = unique_val
                    rows.append(new_row)
            else:
                rows.append(row)

        df = pd.DataFrame(rows)
        if export_to.endswith('.csv'):
            df.to_csv(export_to, index=False)
        elif export_to.endswith('.tsv'):
            df.to_csv(export_to, sep='\t', index=False)
        else:
            raise ValueError(f'Export path must end in .json, .csv, or .tsv. Received {export_to}.')

    def assign_folds(self, num_folds, test_frac=0.2, val_frac=None, stratify=True, predefined_test_ids=None, method='kfold', aggregate_by=None, seed=None):
        '''
        Assigns samples to train, test, and val sets. Test set is identical across all folds, while train and val sets are unique to each fold.
        Val sets can be assigned using either StratifiedKFold or Monte-Carlo sampling. If StratifiedKFold, every non-test sample is assigned to a val set exactly once.

        Args:
            num_folds (int): Number of folds to assign.
            test_frac (float): Fraction of data to assign to test set.
            val_frac (float): Fraction of data to assign to val set. Must be specified if num_folds == 1 or method == 'monte-carlo'.
            stratify (bool): If True, will stratify by all label attributes. If False, will not stratify.
            predefined_test_ids (list): List of ids to manually assign to test set. If specified, test_frac is ignored.
            method (str): Must be 'kfold' or 'monte-carlo'. If 'kfold', will use StratifiedKFold to assign validation folds. If 'monte-carlo', will sample independent random train-test splits. Note that method doesn't matter if num_folds == 1.
            aggregate_by (str): Attribute to aggregate by before splitting. If specified, will aggregate by this attribute before splitting. For example, if aggregate_by = 'case_id', will ensure that all samples from the same case are in the same fold.
            seed (int): Random seed for train-test split.
        '''
        if stratify:
            print(f'Assigning {num_folds} folds to split, stratified by {list(self.data[0]["labels"].keys())}.')
        else:
            print(f'Assigning {num_folds} folds to split, not stratified.')

        if aggregate_by is not None:
            print(f'Aggregating samples by {aggregate_by} before splitting.')
                    
        if num_folds == 1 or method == 'monte-carlo':
            assert val_frac is not None, 'val_frac must be specified if num_folds == 1 or method == "monte-carlo".'
        else:
            assert val_frac is None, 'val_frac must be None if num_folds > 1 and method == "kfold".'

        # Randomly reorder samples
        random.seed(seed)
        random.shuffle(self.data)

        # Aggregate data by the specified attribute
        if aggregate_by:
            agg_id_to_sample_ids = {}
            for sample in self.data:
                agg_ids = sample[aggregate_by]
                if isinstance(agg_ids, str):
                    agg_ids = [agg_ids]
                assert len(set(agg_ids)) == 1, f'Aggregate attribute {aggregate_by} must have a single unique value per sample. Received {agg_ids}.'
                agg_id = agg_ids[0]
                if agg_id not in agg_id_to_sample_ids:
                    agg_id_to_sample_ids[agg_id] = []
                agg_id_to_sample_ids[agg_id].append(sample['id'])
        else:
            agg_id_to_sample_ids = {sample['id']: [sample['id']] for sample in self.data} # If not further aggregating, each sample is its own aggregate

        # Get labels for stratification
        if stratify:
            id_to_labels = {sample['id']: ''.join(str(label) for label in sample['labels'].values()) for sample in self.data}
            agg_id_to_labels = {}
            for agg_id, sample_ids in agg_id_to_sample_ids.items():
                assert len(set(id_to_labels[sample_id] for sample_id in sample_ids)) == 1, f'Trying to aggregate by {aggregate_by}, but found inconsistent labels for samples with aggregator attribute {agg_id}.'
                agg_id_to_labels[agg_id] = id_to_labels[sample_ids[0]]
        else:
            agg_id_to_labels = {agg_id: 0 for agg_id in agg_id_to_sample_ids.keys()} # Using 0 as a dummy label

        # Get ids of test partition
        if predefined_test_ids is not None:
            test_ids = predefined_test_ids
            train_val_agg_ids = []
            for agg_id, sample_ids in agg_id_to_sample_ids.items():
                # Make sure that all samples from the same aggregate are either in the test set or not in the test set
                if all(sample_id not in test_ids for sample_id in sample_ids):
                    train_val_agg_ids.append(agg_id)
                elif all(sample_id in test_ids for sample_id in sample_ids):
                    pass
                else:
                    raise ValueError(f'Predefined test IDs are incompatible with aggregator attribute {agg_id}.')
            print(f'    Manually assigned {len(test_ids)} test samples out of {len(self.data)}.')
        elif test_frac == 1:
            test_ids = list(set([sample['id'] for sample in self.data]))
            train_val_agg_ids = []
            print(f'    Assigned all {len(test_ids)} samples to test.')
        elif test_frac == 0:
            train_val_agg_ids = list(agg_id_to_labels.keys())
            test_ids = []
            print(f'    Assigned zero samples to test.')
        else:
            # Split based on aggregate ids
            train_val_agg_ids, test_agg_ids = compute_split(list(agg_id_to_labels.keys()), test_size=int(test_frac * len(agg_id_to_labels)), random_state=seed, stratify=list(agg_id_to_labels.values()) if stratify else None)
            test_ids = [id for agg_id in test_agg_ids for id in agg_id_to_sample_ids[agg_id]]
            print(f'    Assigned {len(test_ids)} test samples out of {len(self.data)}.')

        # Get ids of val partition
        if val_frac == 0:
            val_ids = {f'fold_{fold}': [] for fold in range(num_folds)}
            print(f'    Assigned zero samples to val out of remaining {len(self.data) - len(test_ids)} samples.')
        elif num_folds == 1:
            train_agg_ids, val_agg_ids = compute_split(train_val_agg_ids, test_size=int(val_frac * len(agg_id_to_labels)), random_state=seed, stratify=[agg_id_to_labels[agg_id] for agg_id in train_val_agg_ids] if stratify else None)
            val_ids = {'fold_0': [id for agg_id in val_agg_ids for id in agg_id_to_sample_ids[agg_id]]}
            print(f'    Assigned {len(val_ids["fold_0"])} val samples out of remaining {len(self.data) - len(test_ids)} samples.')
        else:
            val_ids = {}
            if method == 'monte-carlo':
                for fold in range(num_folds):
                    new_seed = seed + fold  # Generate a new seed for each fold for reproducible sampling
                    train_agg_ids, val_agg_ids = compute_split(train_val_agg_ids, test_size=int(val_frac * len(agg_id_to_labels)), random_state=new_seed, stratify=[agg_id_to_labels[agg_id] for agg_id in train_val_agg_ids] if stratify else None)
                    val_ids[f'fold_{fold}'] = [id for agg_id in val_agg_ids for id in agg_id_to_sample_ids[agg_id]]
                # print(f'    Assigned {len(val_ids["fold_0"])} val samples to each fold out of remaining {len(self.data) - len(test_ids)} using Monte-Carlo sampling.')
                print(f'    Train: {len(self.data) - len(test_ids) - len(val_ids["fold_0"])} samples | Val: {len(val_ids["fold_0"])} samples | Test: {len(test_ids)} samples | Sampling: Monte-Carlo')
            elif method == 'kfold':
                for fold, (train_index, val_index) in enumerate(StratifiedKFold(num_folds).split(train_val_agg_ids, [agg_id_to_labels[agg_id] for agg_id in train_val_agg_ids])):
                    val_ids[f'fold_{fold}'] = [id for i in val_index for id in agg_id_to_sample_ids[train_val_agg_ids[i]]]
                # print(f'    Assigned {len(val_ids["fold_0"])} val samples to each fold out of remaining {len(self.data) - len(test_ids)} using StratifiedKFold.')
                print(f'    Train: {len(self.data) - len(test_ids) - len(val_ids["fold_0"])} samples | Val: {len(val_ids["fold_0"])} samples | Test: {len(test_ids)} samples | Sampling: StratifiedKFold')
            else:
                raise ValueError(f'Sampling method must be "kfold" or "monte-carlo". Received {method}.')

        # Transfer partition assignments to self.data
        for sample in self.data:
            sample['folds'] = []
            for fold_idx in range(num_folds):
                if sample['id'] in test_ids:
                    sample['folds'].append('test')
                elif sample['id'] in val_ids[f'fold_{fold_idx}']:
                    sample['folds'].append('val')
                else:
                    sample['folds'].append('train')

        # Sort so that all the test ids appear first, then all the val ids in fold 0, then all the val ids in fold 1, etc.
        self.data.sort(key=lambda x: x['folds'] + [x['id']])
        self.num_folds = num_folds

    def remove_all_folds(self):
        '''
        Removes fold assignments from the split.
        '''
        print('Removing all fold assignments.')
        for sample in self.data:
            sample['folds'] = []
        self.num_folds = 0

    def replace_folds(self, replace_from, replace_to, selected_ids = None, selected_folds = None):
        '''
        Replaces fold assignments in the split.

        Args:
            replace_from (str): Fold assignment to replace
            replace_to (str): Fold assignment to replace with
            selected_ids (list): List of sample IDs to replace fold assignments for. If None, will replace fold assignments for all samples.
            selected_folds (list): List of fold indices to replace fold assignments for. If None, will replace fold assignments for all folds.
        '''
        
        reassigned_samples_per_fold = {fold_idx: 0 for fold_idx in range(self.num_folds)}
        for sample in self.data:
            if selected_ids is not None and sample['id'] not in selected_ids:
                continue
            for fold_idx, current_assignment in enumerate(sample['folds']):
                if selected_folds is not None and fold_idx not in selected_folds:
                    continue
                if current_assignment == replace_from:
                    sample['folds'][fold_idx] = replace_to
                    reassigned_samples_per_fold[fold_idx] += 1
        
        for fold_idx, num_reassigned in reassigned_samples_per_fold.items():
            if num_reassigned > 0:
                print(f'Reassigned {num_reassigned} samples in fold {fold_idx} from {replace_from} to {replace_to}.')

    def sample_fewshots(self, num_shots, label_attr, num_bootstraps = 1, seed = 42):
        '''
        Samples a few-shot dataset from the split, randomly sampling num_shots samples per class from the training set of each fold (without replacement).
        All non-selected samples in each fold are indicated by setting their fold column to '-1'. 
        Note that this method assumes that folds have already been assigned to the split.

        Args:
            num_shots (int): Number of shots per class
            label_attr (str): Name of the label attribute to use for few-shot sampling
            num_bootstraps (int): Number of bootstrap folds to generate from a single fold. If >1, will sample multiple times from the same fold.
            seed (int): Random seed
        '''
        # Check that folds have been assigned
        assert self.num_folds > 0, 'Folds have not been assigned. Please run self.assign_folds() method first.'

        # Create bootstrap folds if necessary
        if num_bootstraps > 1:
            assert self.num_folds == 1, 'Bootstrap sampling is only supported for single-fold splits.'
            # Duplicate the fold num_bootstraps times
            for sample in self.data:
                sample['folds'] = sample['folds'] * num_bootstraps
            self.num_folds = num_bootstraps

        # Get the unique values of the label attribute
        unique_classes = set([sample['labels'][label_attr] for sample in self.data])
        print(f'Sampling {num_shots} training examples per class for label {label_attr}, with {len(unique_classes)} classes, for a total of {num_shots * len(unique_classes)} examples.')

        # Set the fold to '-1' for all training samples
        self.replace_folds('train', '-1')

        # Sample num_shots samples per class from the training set of each fold
        np.random.seed(seed)
        for fold in range(self.num_folds):
            for label in unique_classes:
                train_ids_with_current_label = [sample['id'] for sample in self.data if sample['folds'][fold] == '-1' and sample['labels'][label_attr] == label]
                
                # Select num_shots samples per class
                if len(train_ids_with_current_label) < num_shots:
                    selected_examples = train_ids_with_current_label
                    print(f'\033[91mWARNING:\033[0m Class {label} in fold {fold} has only {len(selected_examples)} samples, which is less than the requested {num_shots} shots.')
                else:
                    selected_examples = np.random.choice(train_ids_with_current_label, num_shots, replace=False)

                # Set the fold to 'train' for the selected samples
                self.replace_folds('-1', 'train', selected_examples, selected_folds = [fold])
    
    def visualize_folds(self, saveto = None):
        '''
        Visualizes the split and optionally saves the plot.
        
        Args:
            saveto (str): Path to save the plot. If None, will display the plot.
        '''
        # Check that folds have been assigned
        assert self.num_folds > 0, 'Folds have not been assigned. Please run self.assign_kfolds() method first.'
        
        # Sort so that all the test ids appear first, then all the val ids in fold 0, then all the val ids in fold 1, etc.
        data_to_plot = sorted(self.data, key=lambda x: x['folds'] + [x['id']])

        #### Visualize split
        colors = {'train': 'blue', 'val': 'yellow', 'test': 'red', '-1': 'gray'}
        plt.figure(figsize=(10, 1 + 0.2*self.num_folds))

        for fold_idx in tqdm(range(self.num_folds), desc='Visualizing split'):
            # Iterate through ids
            for j, sample in enumerate(data_to_plot):
                # Get the color for this fold
                color = colors[sample['folds'][fold_idx]]
                # Get the y value for this id
                y = fold_idx
                # Plot a point for this id
                plt.plot(j, y, marker='|', color=color, linestyle='')

        plt.yticks(range(self.num_folds), [f'fold_{i}' for i in range(self.num_folds)])
        plt.xlabel('Sample Index')
        plt.ylabel('fold')
        plt.tight_layout()  # Ensure the x-ticks are fully visible

        # Create custom artists for legend
        test_artist = plt.Line2D((0,1),(0,0), color='red', marker='o', linestyle='')
        val_artist = plt.Line2D((0,1),(0,0), color='yellow', marker='o', linestyle='')
        train_artist = plt.Line2D((0,1),(0,0), color='blue', marker='o', linestyle='')
        plt.legend([train_artist, val_artist, test_artist],['Train', 'Validation', 'Test'], loc='upper right')
        
        if saveto is not None:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            plt.savefig(saveto)
        else:
            plt.show()
        plt.close()

    def visualize_labels(self, fold = 0, saveto = None):
        '''
        Visualizes the label distribution for a fold and optionally saves the plot.
        
        Args:
            fold (int): Fold number to visualize. Defaults to 0.
            saveto (str): Path to save the plot. If None, will display the plot.
        '''
        # Sort so that all the test ids appear first, then all the val ids in fold 0, then all the val ids in fold 1, etc.
        data_to_plot = sorted(self.data, key=lambda x: x['folds'] + [x['id']])
        label_attrs = list(data_to_plot[0]['labels'].keys())
        fig, axs = plt.subplots(1, len(label_attrs), figsize=(6 * len(label_attrs), 4))
        if len(label_attrs) == 1:
            axs = [axs]

        colors = {'train': 'blue', 'val': 'yellow', 'test': 'red', '-1': 'gray'}

        for idx, label_attr in enumerate(label_attrs):
            ax = axs[idx]
            modes = set([sample['folds'][fold] for sample in data_to_plot])

            # Initialize a dictionary to hold the counts of each label per mode
            label_counts = {mode: {} for mode in modes}
            all_labels = set()

            # Aggregate counts of each label for each mode
            for mode in modes:
                labels = [sample['labels'][label_attr] for sample in data_to_plot if sample['folds'][fold] == mode]
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_counts[mode] = dict(zip(unique_labels, counts))
                all_labels.update(unique_labels)

            # Convert the label set to a sorted list
            all_labels = sorted(all_labels)
            x = np.arange(len(all_labels))  # the label locations
            width = 0.2  # the width of the bars

            # Plot each mode's label counts as grouped bars
            for i, mode in enumerate(modes):
                counts = [label_counts[mode].get(label, 0) for label in all_labels]
                frac = np.array(counts) / np.sum(counts) if np.sum(counts) else 0
                ax.bar(x + i*width, frac, width, label=mode, color=colors[mode])

                # Annotate bar with counts, horizontally
                for j, count in enumerate(counts):
                    ax.text(x[j] + i*width, frac[j] + 0.01, count, ha='center', va='bottom', rotation=90)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_xlabel(label_attr)
            ax.set_ylabel('Fraction of samples')
            ax.set_title(f'Label distribution in fold {fold} ({label_attr})')
            ax.set_xticks(x + width / len(modes))
            ax.set_xticklabels(all_labels)
            ax.legend()

        plt.tight_layout()
        if saveto is not None:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            plt.savefig(saveto)
        else:
            plt.show()
        plt.close()

    def get_label_distribution(self, label_attr):
        '''
        Returns the label distribution across all modes (train, val, and test).

        Args:
            label_attr (str): Name of the label attribute to use for label distribution
        '''
        # Get the unique values of the label attribute
        unique_classes = self.unique_classes(label_attr)
        label_distribution = {label: 0 for label in unique_classes}

        # Count the number of samples with each label
        for sample in self.data:
            label_distribution[sample['labels'][label_attr]] += 1

        return label_distribution
    
    def get_sizes(self, fold):
        '''
        Returns the number of samples in each fold.

        Args:
            fold (int): Fold number to get sizes for

        Returns:
            sizes (dict): Number of samples in each mode for the specified fold
        '''
        sizes = {}
        for mode in ['train', 'val', 'test']:
            sizes[mode] = len(self.get_ids(fold, mode))
        return sizes
    
    def get_ids(self, fold, mode = None):
        '''
        Returns the IDs of samples in a given fold and mode.

        Args:
            fold (int): Fold number to get IDs for.
            mode (str, optional): Mode to get IDs for. If not provided, returns IDs for all modes.

        Returns:
            List[str]: List of sample IDs.
        '''
        if mode is None:
            return [sample['id'] for sample in self.data]
        else:
            return [sample['id'] for sample in self.data if sample['folds'][fold] == mode]
    
    def get_ratios(self, fold):
        '''
        Returns the ratio of samples in each fold.
        
        Args:
            fold (int): Fold number to get ratios for
            
        Returns:
            ratios (dict): Ratio of samples in each mode for the specified fold
        '''
        sizes = self.get_sizes(fold)
        total = sum(sizes.values())
        ratios = {mode: f'{(size / total):.2f}' for mode, size in sizes.items()}
        return ratios
    
    def unique_classes(self, label_attr):
        '''
        Returns the unique classes for a given label attribute.

        Args:
            label_attr (str): Name of the label attribute to use
        '''
        return set(self.y(label_attr))
    
    def y(self, label_attr, fold = None, mode = None):
        '''
        Returns all labels for a given label attribute, fold, and mode.

        Args:
            label_attr (str): Name of the label attribute to use
            fold (int): Fold number to subset modes for. If not provided, will get labels for all modes.
            mode (str): Mode to get labels for. If not provided, will get labels for all modes.

        Returns:
            labels (np.array): Array of labels (shape: (num_samples,))
        '''
        # Assert that either both or none of fold and mode are provided
        assert (fold is not None and mode is not None) or (fold is None and mode is None), 'Either both or none of fold and mode must be provided.'

        if fold is not None and mode is not None:
            return np.array([sample['labels'][label_attr] for sample in self.data if sample['folds'][fold] == mode])
        else:
            return np.array([sample['labels'][label_attr] for sample in self.data])