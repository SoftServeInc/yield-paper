from argparse import Namespace
import csv
from logging import Logger
import pickle
import random
from typing import List, Set, Tuple, Union
import os

import numpy as np
from tqdm import tqdm

from .data import MoleculeDatapoint, MoleculeDataset, ReactionDatapoint, ReactionDataset
from .scaffold import log_scaffold_stats, scaffold_split
from chemprop.features import load_features
from chemprop.mol_utils import str_to_mol


def get_task_names(path: str, use_compound_names: bool = False, reaction: bool = False) -> List[str]:
    """
    Gets the task names from a data CSV file.

    :param path: Path to a CSV file.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :param reaction: Whether file contains reactions instead of molecules.
    :return: A list of task names.
    """
    index = 1 + use_compound_names + reaction
    task_names = get_header(path)[index:]

    return task_names


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_num_tasks(path: str) -> int:
    """
    Gets the number of tasks in a data CSV file.

    :param path: Path to a CSV file.
    :return: The number of tasks.
    """
    return len(get_header(path)) - 1


def get_smiles(path: str, header: bool = True, reaction: bool = False) -> Union[List[str], List[Tuple[str, str]]]:
    """
    Returns the smiles strings from a data CSV file (assuming the first line is a header).

    :param path: Path to a CSV file.
    :param header: Whether the CSV file contains a header (that will be skipped).
    :param reaction: Whether the CSV file contains reactions instead of molecules.
    :return: A list of smiles strings.
    """
    with open(path) as f:
        reader = csv.reader(f)
        if header:
            next(reader)  # Skip header
        if reaction:
            smiles = [(line[0], line[1]) for line in reader]
        else:
            smiles = [line[0] for line in reader]

    return smiles


def filter_invalid_smiles(data: Union[MoleculeDataset, ReactionDataset]) -> Union[MoleculeDataset, ReactionDataset]:
    """
    Filters out invalid SMILES.

    :param data: A MoleculeDataset or ReactionDataset.
    :return: A MoleculeDataset or ReactionDataset with only valid molecules.
    """
    if isinstance(data, ReactionDataset):
        # Allow molecules without heavy atoms
        return ReactionDataset([datapoint for datapoint in data
                                if datapoint.rsmiles != '' and datapoint.rmol is not None
                                if datapoint.psmiles != '' and datapoint.pmol is not None
                                ])
    else:
        return MoleculeDataset([datapoint for datapoint in data
                                if datapoint.smiles != '' and datapoint.mol is not None
                                and datapoint.mol.GetNumHeavyAtoms() > 0])


def get_data(path: str,
             skip_invalid_smiles: bool = True,
             args: Namespace = None,
             reaction: bool = None,
             features_path: List[str] = None,
             max_data_size: int = None,
             use_compound_names: bool = None,
             logger: Logger = None) -> Union[MoleculeDataset, ReactionDataset]:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.

    :param path: Path to a CSV file.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param args: Arguments.
    :param reaction: Whether loading reactions instead of molecules.
    :param features_path: A list of paths to files containing features. If provided, it is used
    in place of args.features_path.
    :param max_data_size: The maximum number of data points to load.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :param logger: Logger.
    :return: A MoleculeDataset or ReactionDataset containing smiles strings and target values along
    with other info such as additional features and compound names when desired.
    """
    debug = logger.debug if logger is not None else print

    if args is not None:
        # Prefer explicit function arguments but default to args if not provided
        features_path = features_path if features_path is not None else args.features_path
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        use_compound_names = use_compound_names if use_compound_names is not None else args.use_compound_names
        reaction = reaction if reaction is not None else args.reaction
    else:
        use_compound_names = False
        reaction = reaction if reaction is not None else False

    # Load features
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    skip_smiles = set()

    # Load data
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        lines = []
        for line in reader:
            rsmiles = line[0]
            psmiles = line[1] if reaction else None

            if rsmiles in skip_smiles or psmiles in skip_smiles:
                continue

            lines.append(line)

        keep_idxs = list(range(len(lines)))
        if max_data_size is not None:
            random.seed(args.seed)
            random.shuffle(keep_idxs)
            keep_idxs = keep_idxs[:max_data_size]

        if reaction:
            data = ReactionDataset([
                ReactionDatapoint(
                    line=lines[i],
                    args=args,
                    features=features_data[i] if features_data is not None else None,
                    use_compound_names=use_compound_names
                ) for i in tqdm(keep_idxs, total=len(keep_idxs))
            ])
        else:
            data = MoleculeDataset([
                MoleculeDatapoint(
                    line=lines[i],
                    args=args,
                    features=features_data[i] if features_data is not None else None,
                    use_compound_names=use_compound_names
                ) for i in tqdm(keep_idxs, total=len(keep_idxs))
            ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)

    return data


def get_data_from_smiles(smiles: Union[List[str], List[Tuple[str, str]]],
                         skip_invalid_smiles: bool = True,
                         args: Namespace = None,
                         logger: Logger = None) -> Union[MoleculeDataset, ReactionDataset]:
    """
    Converts SMILES to a MoleculeDataset or ReactionDataset.

    :param smiles: A list of (tuples of) SMILES strings.
    :param args: Arguments.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param logger: Logger.
    :return: A MoleculeDataset with all of the provided SMILES.
    """
    debug = logger.debug if logger is not None else print

    if isinstance(smiles[0], str):
        data = MoleculeDataset([MoleculeDatapoint(line=[smile], args=args) for smile in smiles])
    else:
        data = ReactionDataset([ReactionDatapoint(line=list(smile), args=args) for smile in smiles])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def split_data(data: Union[MoleculeDataset, ReactionDataset],
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               args: Namespace = None,
               logger: Logger = None) -> Union[Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset],
                                               Tuple[ReactionDataset, ReactionDataset, ReactionDataset]]:
    """
    Splits data into training, validation, and test splits.

    :param data: A MoleculeDataset or ReactionDataset.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param seed: The random seed to use before shuffling data.
    :param args: Namespace of arguments.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    if args is not None:
        folds_file, val_fold_index, test_fold_index = \
            args.folds_file, args.val_fold_index, args.test_fold_index
    else:
        folds_file = val_fold_index = test_fold_index = None
    
    if split_type == 'crossval':
        index_set = args.crossval_index_sets[args.seed]
        data_split = []
        for split in range(3):
            split_indices = []
            for index in index_set[split]:
                with open(os.path.join(args.crossval_index_dir, f'{index}.pkl'), 'rb') as rf:
                    split_indices.extend(pickle.load(rf))
            data_split.append([data[i] for i in split_indices])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)
    
    elif split_type == 'index_predetermined':
        split_indices = args.crossval_index_sets[args.seed]
        assert len(split_indices) == 3
        data_split = []
        for split in range(3):
            data_split.append([data[i] for i in split_indices[split]])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'predetermined':
        if not val_fold_index:
            assert sizes[2] == 0  # test set is created separately so use all of the other data for train and val
        assert folds_file is not None
        assert test_fold_index is not None

        try:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f)
        except UnicodeDecodeError:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f, encoding='latin1')  # in case we're loading indices from python2
        # assert len(data) == sum([len(fold_indices) for fold_indices in all_fold_indices])

        log_scaffold_stats(data, all_fold_indices, logger=logger)

        folds = [[data[i] for i in fold_indices] for fold_indices in all_fold_indices]

        test = folds[test_fold_index]
        if val_fold_index is not None:
            val = folds[val_fold_index]

        train_val = []
        for i in range(len(folds)):
            if i != test_fold_index and (val_fold_index is None or i != val_fold_index):
                train_val.extend(folds[i])

        if val_fold_index is not None:
            train = train_val
        else:
            random.seed(seed)
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

        if args.reaction:
            return ReactionDataset(train), ReactionDataset(val), ReactionDataset(test)
        else:
            return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)
    
    elif split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, seed=seed, logger=logger)

    elif split_type == 'random':
        data.shuffle(seed=seed)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]

        if args.reaction:
            return ReactionDataset(train), ReactionDataset(val), ReactionDataset(test)
        else:
            return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    else:
        raise ValueError(f'split_type "{split_type}" not supported.')


def get_class_sizes(data: Union[MoleculeDataset, ReactionDataset]) -> List[List[float]]:
    """
    Determines the proportions of the different classes in the classification dataset.

    :param data: A classification dataset
    :return: A list of lists of class proportions. Each inner list contains the class proportions
    for a task.
    """
    targets = data.targets()

    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if targets[i][task_num] is not None:
                valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:
        # Make sure we're dealing with a binary classification task
        assert set(np.unique(task_targets)) <= {0, 1}

        try:
            ones = np.count_nonzero(task_targets) / len(task_targets)
        except ZeroDivisionError:
            ones = float('nan')
            print('Warning: class has no targets')
        class_sizes.append([1 - ones, ones])

    return class_sizes


def validate_data(data_path: str) -> Set[str]:
    """
    Validates a data CSV file, returning a set of errors.

    :param data_path: Path to a data CSV file.
    :return: A set of error messages.
    """
    errors = set()

    header = get_header(data_path)

    with open(data_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        smiles, targets = [], []
        for line in reader:
            smiles.append(line[0])
            targets.append(line[1:])

    # Validate header
    if len(header) == 0:
        errors.add('Empty header')
    elif len(header) < 2:
        errors.add('Header must include task names.')

    mol = str_to_mol(header[0])
    if mol is not None:
        errors.add('First row is a SMILES string instead of a header.')

    # Validate smiles
    for smile in tqdm(smiles, total=len(smiles)):
        mol = str_to_mol(smile)
        if mol is None:
            errors.add('Data includes an invalid SMILES.')

    # Validate targets
    num_tasks_set = set(len(mol_targets) for mol_targets in targets)
    if len(num_tasks_set) != 1:
        errors.add('Inconsistent number of tasks for each molecule.')

    if len(num_tasks_set) == 1:
        num_tasks = num_tasks_set.pop()
        if num_tasks != len(header) - 1:
            errors.add('Number of tasks for each molecule doesn\'t match number of tasks in header.')

    unique_targets = set(np.unique([target for mol_targets in targets for target in mol_targets]))

    if unique_targets <= {''}:
        errors.add('All targets are missing.')

    for target in unique_targets - {''}:
        try:
            float(target)
        except ValueError:
            errors.add('Found a target which is not a number.')

    return errors
