from argparse import Namespace
from typing import List, Union
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np

from .mpn import MPN, MPNDiff
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights


class Model(nn.Module):
    """Base class for molecule and reaction models, which only differ in their encoding."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the Model.

        :param classification: Whether the model is a classification model.
        """
        super(Model, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace):
        """
        Creates the encoder for the model.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.diff_hidden_size if args.reaction else args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                # dropout,
                # nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        ffn2 = [nn.Dropout(args.dropout), nn.Linear(args.ffn_hidden_size, args.output_size)]
        self.ffn2 = nn.Sequential(*ffn2)

    def forward(self, *input):
        """
        Defines the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError


class MoleculeModel(Model):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__(classification, multiclass)

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

        if args.freeze_mpn:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        output = self.ffn(self.encoder(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output


class ReactionModel(Model):
    """
    A ReactionModel is a model which contains the same message passing
    network for the reactant and product molecules, followed by the
    formation of difference features and subsequent encoding with a
    difference message passing network, followed by feed-forward layers.
    """

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the ReactionModel.

        :param classification: Whether the model is a classification model.
        """
        super(ReactionModel, self).__init__(classification, multiclass)

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoders for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args, return_atom_hiddens=True)
        self.diff_encoder = MPNDiff(args, atom_fdim=args.hidden_size)

        if args.freeze_mpn:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if args.freeze_mpn_diff:
            for param in self.diff_encoder.parameters():
                param.requires_grad = False

    def forward(self,
                rbatch: Union[List[str], BatchMolGraph],
                pbatch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Runs the ReactionModel on input.

        :param rbatch: A list of SMILES strings or a BatchMolGraph for the reactants.
        :param pbatch: A list of SMILES strings or a BatchMolGraph for the products.
        :param features_batch: A list of ndarrays containing additional features.
        :return: The output of the ReactionModel.
        """
        r_atom_features = self.encoder(rbatch)
        p_atom_features = self.encoder(pbatch)

        diff_features = p_atom_features - r_atom_features

        output, num_not_zero_diff, middle = self.diff_encoder(diff_features, pbatch, features_batch)

        output = self.ffn(output)  # use product graph
        after_diff_encoder_layer = output.detach().data.cpu().numpy()
        output = self.ffn2(output)
        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape(
                (output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(
                    output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output, num_not_zero_diff, middle, after_diff_encoder_layer


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel or ReactionModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel/ReactionModel containing the MPN encoder
             along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    if args.reaction:
        model = ReactionModel(classification=args.dataset_type == 'classification',
                              multiclass=args.dataset_type == 'multiclass')
    else:
        model = MoleculeModel(classification=args.dataset_type == 'classification',
                              multiclass=args.dataset_type == 'multiclass')
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
