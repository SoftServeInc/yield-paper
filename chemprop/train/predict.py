from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from chemprop.data import MoleculeDataset, ReactionDataset, StandardScaler

import collections

def predict(model: nn.Module,
            data: Union[MoleculeDataset, ReactionDataset],
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset or ReactionDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []

    num_changed_atoms = []

    num_iters, iter_step = len(data), batch_size

    for i in trange(0, num_iters, iter_step):
        # Prepare batch
        data_batch = data[i:i + batch_size]
        mol_batch = ReactionDataset(data_batch) if isinstance(data, ReactionDataset) else MoleculeDataset(data_batch)
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        with torch.no_grad():
            if isinstance(data, ReactionDataset):
                rbatch, pbatch = list(zip(*batch))
                batch_preds, num_not_zero_diff = model(rbatch, pbatch, features_batch)

            else:
                batch_preds = model(batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
        if isinstance(data, ReactionDataset):

            changed_atoms = np.array([c[0] for c in num_not_zero_diff])
            num_changed_atoms.extend(changed_atoms.tolist())


    return preds, num_changed_atoms
