from argparse import Namespace
import random
from typing import Callable, List, Tuple, Union

import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset
from rdkit import Chem

from .scaler import StandardScaler
from chemprop.features import get_features_generator
from chemprop.mol_utils import str_to_mol


class Datapoint:
    """Base class for other Datapoint classes."""

    def set_features(self, features: np.ndarray):
        """
        Sets the features.

        :param features: A 1-D numpy array of features.
        """
        self.features = features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[float]):
        """
        Sets the targets.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets


class MoleculeDatapoint(Datapoint):
    """A MoleculeDatapoint contains a single molecule and its associated features and targets."""

    def __init__(self,
                 line: List[str],
                 args: Namespace = None,
                 features: np.ndarray = None,
                 use_compound_names: bool = False):
        """
        Initializes a MoleculeDatapoint, which contains a single molecule.

        :param line: A list of strings generated by separating a line in a data CSV file by comma.
        :param args: Arguments.
        :param features: A numpy array containing additional features (ex. Morgan fingerprint).
        :param use_compound_names: Whether the data CSV includes the compound name on each line.
        """
        if args is not None:
            self.features_generator = args.features_generator
            self.args = args
        else:
            self.features_generator = self.args = None

        if features is not None and self.features_generator is not None:
            raise ValueError('Currently cannot provide both loaded features and a features generator.')

        self.features = features

        if use_compound_names:
            self.compound_name = line[0]  # str
            line = line[1:]
        else:
            self.compound_name = None

        self.smiles = line[0]  # str
        self.mol = str_to_mol(self.smiles, explicit_hydrogens=args.explicit_hydrogens if args is not None else False)

        # Generate additional features if given a generator
        if self.features_generator is not None:
            self.features = []

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                if self.mol is not None and self.mol.GetNumHeavyAtoms() > 0:
                    self.features.extend(features_generator(self.mol))

            self.features = np.array(self.features)

        # Fix nans in features
        if self.features is not None:
            replace_token = 0
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Create targets
        self.targets = [float(x) if x != '' else None for x in line[1:]]


class ReactionDatapoint(Datapoint):
    """A ReactionDatapoint contains a single reaction (two molecules) and its associated features and targets."""

    def __init__(self,
                 line: List[str],
                 args: Namespace = None,
                 features: np.ndarray = None,
                 use_compound_names: bool = False):
        """
        Initializes a ReactionDatapoint, which contains a single reaction.

        :param line: A list of strings generated by separating a line in a data CSV file by comma.
        :param args: Arguments.
        :param features: A numpy array containing additional features (ex. Morgan fingerprint).
        :param use_compound_names: Whether the data CSV includes the compound names on each line.
        """
        if args is not None:
            self.features_generator = args.features_generator
            self.args = args
        else:
            self.features_generator = self.args = None

        if features is not None and self.features_generator is not None:
            raise ValueError('Currently cannot provide both loaded features and a features generator.')

        self.features = features

        if use_compound_names:
            self.compound_name = line[0] + '>' + line[1]  # str
            line = line[2:]
        else:
            self.compound_name = None

        self.rsmiles = line[0]  # str
        self.psmiles = line[1]  # str
        self.rmol = str_to_mol(self.rsmiles, explicit_hydrogens=args.explicit_hydrogens if args is not None else False)
        self.pmol = str_to_mol(self.psmiles, explicit_hydrogens=args.explicit_hydrogens if args is not None else False)

        # Generate additional features if given a generator
        if self.features_generator is not None:
            self.features = []

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                if self.rmol is not None and self.pmol is not None:
                    # Use difference features
                    diff_feat = np.asarray(features_generator(self.pmol)) - np.asarray(features_generator(self.rmol))
                    self.features.extend(diff_feat)

            self.features = np.array(self.features)

        # Fix nans in features
        if self.features is not None:
            replace_token = 0
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Create targets
        self.targets = [float(x) if x != '' else None for x in line[2:]]


class Dataset(TorchDataset):
    """Base class for other Dataset classes."""

    def __init__(self, data: List[Datapoint]):
        """
        Initializes a Dataset, which contains a list of Datapoints (e.g. a list of molecules or reactions).

        :param data: A list of Datapoints.
        """
        self.data = data
        self.args = self.data[0].args if len(self.data) > 0 else None
        self.scaler = None

    def smiles(self):
        """
        Returns the smiles strings or tuples thereof associated with the data points.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def mols(self):
        """
        Returns the RDKit molecules or tuples thereof associated with the data points.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def compound_names(self) -> List[str]:
        """
        Returns the compound names associated with the dataset.

        :return: A list of compound names or None if the dataset does not contain compound names.
        """
        if len(self.data) == 0 or self.data[0].compound_name is None:
            return None

        return [d.compound_name for d in self.data]

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each data point (if they exist).

        :return: A list of 1D numpy arrays containing the features for each data point or None if there are no features.
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        return [d.features for d in self.data]

    def targets(self) -> List[List[float]]:
        """
        Returns the targets associated with each data point.

        :return: A list of lists of floats containing the targets.
        """
        return [d.targets for d in self.data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self.data[0].num_tasks() if len(self.data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the features array associated with each data point.

        :return: The size of the features.
        """
        return len(self.data[0].features) if len(self.data) > 0 and self.data[0].features is not None else None

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
    
    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
        Normalizes the features of the dataset using a StandardScaler (subtract mean, divide by standard deviation).

        If a scaler is provided, uses that scaler to perform the normalization. Otherwise fits a scaler to the
        features in the dataset and then performs the normalization.

        :param scaler: A fitted StandardScaler. Used if provided. Otherwise a StandardScaler is fit on
        this dataset and is then used.
        :param replace_nan_token: What to replace nans with.
        :return: A fitted StandardScaler. If a scaler is provided, this is the same scaler. Otherwise, this is
        a scaler fit on this dataset.
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        if scaler is not None:
            self.scaler = scaler

        elif self.scaler is None:
            features = np.vstack([d.features for d in self.data])
            self.scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.scaler.fit(features)

        for d in self.data:
            d.set_features(self.scaler.transform(d.features.reshape(1, -1))[0])

        return self.scaler
    
    def set_targets(self, targets: List[List[float]]):
        """
        Sets the targets for each item in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats containing targets for each data point. This must be the
        same length as the underlying dataset.
        """
        assert len(self.data) == len(targets)
        for i in range(len(self.data)):
            self.data[i].set_targets(targets[i])

    def sort(self, key: Callable):
        """
        Sorts the dataset using the provided key.

        :param key: A function on a Datapoint to determine the sorting order.
        """
        self.data.sort(key=key)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (e.g. the number of molecules or reactions).

        :return: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, item) -> Union[Datapoint, List[Datapoint]]:
        """
        Gets one or more Datapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A Datapoint if an int is provided or a list of Datapoints if a slice is provided.
        """
        return self.data[item]


class MoleculeDataset(Dataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, data: List[MoleculeDatapoint]):
        """
        Initializes a MoleculeDataset, which contains a list of MoleculeDatapoints (i.e. a list of molecules).

        :param data: A list of MoleculeDatapoints.
        """
        super(MoleculeDataset, self).__init__(data)

    def smiles(self) -> List[str]:
        """
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        """
        return [d.smiles for d in self.data]
    
    def mols(self) -> List[Chem.Mol]:
        """
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        """
        return [d.mol for d in self.data]


class ReactionDataset(Dataset):
    """A ReactionDataset contains a list of reactions and their associated features and targets."""
    def __init__(self, data: List[ReactionDatapoint]):
        """
        Initializes a ReactionDataset, which contains a list of ReactionDatapoints (i.e. a list of reactions).

        :param data: A list of ReactionDatapoints.
        """
        super(ReactionDataset, self).__init__(data)

    def smiles(self) -> List[Tuple[str, str]]:
        """
        Returns the tuples of smiles strings associated with the reactions.

        :return: A list of tuples of smiles strings.
        """
        return [(d.rsmiles, d.psmiles) for d in self.data]

    def mols(self) -> List[Tuple[Chem.Mol, Chem.Mol]]:
        """
        Returns the tuples of RDKit molecules associated with the reactions.

        :return: A list of RDKit Mols.
        """
        return [(d.rmol, d.pmol) for d in self.data]
