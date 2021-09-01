from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function


class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int, return_atom_hiddens: bool = False):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param return_atom_hiddens: Return hidden atom feature vectors instead of mol vector.
        """
        super(MPNEncoder, self).__init__()
        self.return_atom_hiddens = return_atom_hiddens
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        if self.depth > 1:
            self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        # print(self.use_input_features)
        if self.use_input_features and not self.return_atom_hiddens:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        if self.return_atom_hiddens:
            return atom_hiddens

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                print('add cached zero vec')
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        
        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1,features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPNDiffEncoder(nn.Module):
    """A message passing neural network for encoding of custom (difference) features."""

    def __init__(self, args: Namespace, atom_fdim: int):
        """Initializes the MPNDiffEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        """
        super(MPNDiffEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = get_bond_fdim(args)
        self.hidden_size = args.diff_hidden_size
        self.bias = args.bias
        self.depth = args.depth_diff
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.use_input_features = args.use_input_features
        self.args = args

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        self.W_i = nn.Linear(self.atom_fdim, self.hidden_size, bias=self.bias)

        # Shared weight matrix across depths (default)
        if self.depth > 1:
            self.W_h = nn.Linear(self.hidden_size + self.bond_fdim, self.hidden_size, bias=self.bias)

        if self.depth > 0:
            self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self,
                atom_features: torch.FloatTensor,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs with custom features.

        :param atom_features: Atom features for the BatchMolGraph.
        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        num_not_zero_diff = []
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_bonds, a2b, a2a = f_bonds.cuda(), a2b.cuda(), a2a.cuda()

        for i, (a_start, a_size) in enumerate(a_scope):
            af = atom_features.narrow(0, a_start, a_size)
            num_not_zero_diff.append([torch.sum((torch.sum(af, dim=1) > 0)).item(), a_size])


        # Input
        input = self.W_i(atom_features)  # num_atoms x atom_fdim
        message = self.act_func(input)  # num_atoms x hidden_size

        if self.depth > 0:
            # Message passing
            for depth in range(self.depth - 1):
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x (bond_fdim (+ atom_fdim_MPN))

                # If using bond messages in MPN, bond features include some atom features,
                # but we only want the pure bond features here
                nei_f_bonds = nei_f_bonds[:, :, -self.bond_fdim:]

                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim

                message = self.W_h(message)
                message = self.act_func(input + message)  # num_atoms x hidden_size
                message = self.dropout_layer(message)  # num_atoms x hidden

            nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden #TODO: why a2a not a2b
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            a_input = torch.cat([atom_features, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
            atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
            atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        else:
            atom_hiddens = self.dropout_layer(message)

        # Readout
        vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                vecs.append(mol_vec)

        vecs = torch.stack(vecs, dim=0)  # (num_samples, hidden_size)
        # middle_layer = deepcopy(vecs)


        if self.use_input_features:
            features_batch = features_batch.to(vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
            vecs = torch.cat([vecs, features_batch], dim=1)  # (num_samples, hidden_size)

        return vecs,  num_not_zero_diff, #middle_layer  # num_samples x hidden


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False,
                 return_atom_hiddens: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        :param return_atom_hiddens: Return hidden atom feature vectors instead of mol vector.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.return_atom_hiddens = return_atom_hiddens
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim, self.return_atom_hiddens)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)

        output = self.encoder.forward(batch, features_batch)

        return output


class MPNDiff(nn.Module):
    """A message passing neural network for encoding of custom (difference) features."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int,
                 graph_input: bool = False):
        """Initializes the MPNDiff.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPNDiff, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim
        self.graph_input = graph_input
        self.encoder = MPNDiffEncoder(self.args, self.atom_fdim)

    def forward(self,
                atom_features: torch.FloatTensor,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings with custom features.

        :param atom_features: Atom features for the batch.
        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)

        output = self.encoder.forward(atom_features, batch, features_batch)

        return output
