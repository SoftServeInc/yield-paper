from rdkit import Chem

RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()


def str_to_mol(string: str, explicit_hydrogens: bool = False) -> Chem.Mol:
    """
    Converts an InChI or SMILES string to an RDKit molecule.

    :param string: The InChI or SMILES string.
    :param explicit_hydrogens: Whether to treat hydrogens explicitly.
    :return: The RDKit molecule.
    """
    if string.startswith('InChI'):
        mol = Chem.MolFromInchi(string, removeHs=not explicit_hydrogens)
    else:
        # Set params here so we don't remove hydrogens with atom mapping
        RDKIT_SMILES_PARSER_PARAMS.removeHs = not explicit_hydrogens
        try:
            mol = Chem.MolFromSmiles(string, RDKIT_SMILES_PARSER_PARAMS)
        except:
            print(f'Invalid smile: {string}')
            return None

    try:
        if explicit_hydrogens:
            return Chem.AddHs(mol)
        else:
            return Chem.RemoveHs(mol)
    except:
        print(f'Invalid smile: {string}')

        return None
