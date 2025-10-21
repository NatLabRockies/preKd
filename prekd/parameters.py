from typing import Dict, List, Union
import pandas as pd


class Parameters:
    def __init__(
        self,
        kfolds: int = 3,
        batch_size: int = 64,
        epochs: int = 5,
        decay: float = 1e-5,
        learning_rate: float = 0.0005,
        atom_features: int = 32,
        bond_features: int = 32,
        mol_features: int = 8,
        num_messages: int = 2,
        dropout: float = 0.05,
        no_mol_frac_weights: bool = False,
        #dense_layers: int = 3,
        prediction_columns: List[str] = None,
        solute_col: str = "solute_smiles",
        solute_feature_cols: List[str] = None,
        solvents_col: str = "solvent_smiles",
        solvent_fracs_col: str = "solvent_mol_fractions", 
        solvent_feature_df: Union[pd.DataFrame, str, None] = None,
        solvent_feature_cols: List[str] = None,
        
        **kwargs
    ):
        """These are all default parameters. They do not guarantee a good model
        generation.

        Parameters
        ----------
        kfolds : int, optional
            Number of folds for cross validation, by default 3
        batch_size : int, optional
            Batch size for training, by default 64
        epochs : int, optional
            Number of epochs for training, by default 5.
        decay : float, optional
            Learning rate decay, by default 1e-5.
        learning_rate : float, optional
            Initial learning rate, by default 0.0005.
        atom_features : int, optional
            Number of atom features, by default 32.
        bond_features : int, optional
            Number of bond features, by default 32.
        mol_features : int, optional
            Number of molecular features, by default 8.
        num_messages : int, optional
            Number of message passing steps, by default 2.
        dropout : float, optional
            Dropout rate, by default 0.05.
        prediction_columns : List[str], optional
            List of columns to be predicted.
        solute_col : str, optional
            Column name for solute SMILES strings, by default "solute_smiles".
        solute_feature_cols : List[str], optional
            List of solute feature columns.
        solvents_col : str, optional
            Column name for solvent SMILES strings, delimited by ';'. 
            Default "solvent_smiles".
        solvent_fracs_col : str, optional
            Column name for solvent fractions, delimited by ';'. 
            Default "solvent_mol_fractions".
        solvent_feature_df : pd.DataFrame or str, optional
            DataFrame (or Path) containing solvent features.
        solvent_feature_cols : List[str], optional
            List of solvent feature columns to be used from solvent_feature_df.

        Can specify additional keyword arguments.
        """
        self.kfolds = kfolds
        self.prediction_columns = prediction_columns
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.mol_features = mol_features
        self.num_messages = num_messages
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.decay = decay
        self.no_mol_frac_weights = no_mol_frac_weights
        self.solute_col = solute_col
        self.solute_feature_cols = solute_feature_cols
        self.solvents_col = solvents_col
        self.solvent_fracs_col = solvent_fracs_col
        self.solvent_feature_df = solvent_feature_df
        self.solvent_feature_cols = solvent_feature_cols

        #if not self.solvents_col:
        #    # default set of solvents
        #    self.solvents_col = ['water', 'ethyl acetate', 'ethanol', 'hexane', 'methanol', 'chloroform',
        #                         'petroleum ether', 'acetonitrile', 'heptane', 'acetone',
        #                         'carbon tetrachloride', 'dichloromethane', 'butanol',
        #                         'methyl tertiary butyl ether', 'isopropanol']

        if self.solvent_feature_df is not None:
            if isinstance(self.solvent_feature_df, str):
                print("Reading solvent feature df from: ", self.solvent_feature_df)
                self.solvent_feature_df = pd.read_csv(self.solvent_feature_df, index_col=0)
            if self.solvent_feature_cols is None:
                self.solvent_feature_cols = self.solvent_feature_df.columns.tolist
                print("Using all solvent features by default: ", self.solvent_feature_cols)
            for col in self.solvents_col:
                assert col in self.solvent_feature_df.index, f"Solvent {col} not found in solvent feature df."

        # Assign any non-default key val pairs
        for key, val in kwargs.items():
            self.__setitem__(key, val)

    @property
    def training_params(self) -> Dict:
        """The training parameter inputs for generating a model.

        Returns
        -------
        Dict
            A dictionary containing the batch size, kfolds, epochs, dropout, decay,
            and learning_rate
        """
        training_params = {
            "batch_size": [self.batch_size],
            "kfolds": list(range(self.kfolds)),
            "epochs": [self.epochs],
            "learning_rate": [self.learning_rate],
            "dropout": [self.dropout],
            "decay": [self.decay],
        }
        return training_params

    def to_dict(self) -> Dict:
        """Returns a dictionary with all the parameters.

        Returns
        -------
        Dict
            Dictionary containing all of the key/val pairs for parameters.
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, param_dict):
        """Generate a Parameters object from a dictionary.

        Parameters
        ----------
        param_dict : Dict
            A dictionary containing key/val pairs of parameters.

        Returns
        -------
        Parameters
            A Parameters class instance.
        """
        return cls(**param_dict)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)
