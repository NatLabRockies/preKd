import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
from typing import List, Union
import ast

tqdm.pandas()


class DoV:
    """
    Class to estimate the distance from the domain of validity based on Morgan fingerprints.
    Supports both solute-only and solute-solvent pair analysis.
    Handles solvent mixtures with mole fractions.
    Expects semicolon-delimited strings for solvent mixtures:
    - solvent_col: "CCO;C;O" (SMILES separated by semicolons)
    - solvent_frac_col: "0.5;0.3;0.2" (fractions separated by semicolons)
    """
    def __init__(self, fingerprint_col="smiles", solvent_col="solvent_smiles", 
                 solvent_frac_col="solvent_fracs", radius=2):
        self.fingerprint_col = fingerprint_col
        self.solvent_col = solvent_col
        self.solvent_frac_col = solvent_frac_col
        self.radius = radius

    def _parse_solvent_list(self, value: Union[str, List[str]]) -> List[str]:
        """Parse solvent SMILES from various formats.
        
        Args:
            value: Can be semicolon-delimited string, list representation, or list
            
        Returns:
            List of SMILES strings
        """
        if isinstance(value, list):
            return value
        
        if isinstance(value, str):
            # Try semicolon-delimited format first
            if ';' in value:
                return [s.strip() for s in value.split(';') if s.strip()]
            # Try list representation
            try:
                result = ast.literal_eval(value)
                if isinstance(result, list):
                    return result
            except:
                pass
            # Single solvent
            return [value]
        
        return [str(value)]

    def _parse_fraction_list(self, value: Union[str, List[float]]) -> List[float]:
        """Parse mole fractions from various formats.
        
        Args:
            value: Can be semicolon-delimited string, list representation, or list
            
        Returns:
            List of floats
        """
        if isinstance(value, list):
            return [float(x) for x in value]
        
        if isinstance(value, str):
            # Try semicolon-delimited format first
            if ';' in value:
                return [float(f.strip()) for f in value.split(';') if f.strip()]
            # Try list representation
            try:
                result = ast.literal_eval(value)
                if isinstance(result, list):
                    return [float(x) for x in result]
            except:
                pass
        # Single fraction
        try:
            return [float(value)]
        except:
            return [1.0]

    def get_fp(self, smiles: str) -> pd.Series:
        """Gets a the fingerprint hashes for a single smiles string.

        Args:
            smiles (str): singe smiles string

        Returns:
            pd.Series: Series containing the hashes for the each fingerprint and a count of their occurance in the molecule.
        """

        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprint(mol, self.radius, useFeatures=False)
        fp = pd.Series(fp.GetNonzeroElements(), name=smiles)
        return fp

    def get_fp_bitvector(self, smiles: str, nbits: int = 2048) -> np.ndarray:
        """Gets a Morgan fingerprint as a bit vector for similarity calculations.

        Args:
            smiles (str): single smiles string
            nbits (int): number of bits in the fingerprint

        Returns:
            np.ndarray: Binary fingerprint vector
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nbits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=nbits)
        return np.array(fp)

    def get_mixture_fp_bitvector(self, smiles_list: Union[str, List[str]], 
                                  fracs: Union[str, List[float]], 
                                  nbits: int = 2048) -> np.ndarray:
        """Gets a weighted Morgan fingerprint for a solvent mixture.
        
        The mixture fingerprint is calculated as a weighted average of individual
        component fingerprints based on mole fractions.

        Args:
            smiles_list (str or list): Semicolon-delimited string ("CCO;C") or list of SMILES
            fracs (str or list): Semicolon-delimited string ("0.7;0.3") or list of fractions
            nbits (int): number of bits in the fingerprint

        Returns:
            np.ndarray: Weighted fingerprint vector (not binary, contains fractional values)
        """
        # Parse inputs
        smiles_list = self._parse_solvent_list(smiles_list)
        fracs = self._parse_fraction_list(fracs)
        
        # Ensure equal length
        if len(fracs) < len(smiles_list):
            # If fewer fractions than SMILES, pad with equal weights
            fracs.extend([1.0] * (len(smiles_list) - len(fracs)))
        elif len(fracs) > len(smiles_list):
            # Truncate fractions if more than SMILES
            fracs = fracs[:len(smiles_list)]
            
        # Normalize fractions
        fracs = np.array(fracs)
        if fracs.sum() > 0:
            fracs = fracs / fracs.sum()
        else:
            fracs = np.ones(len(fracs)) / len(fracs)
        
        # Calculate weighted fingerprint
        weighted_fp = np.zeros(nbits)
        for smiles, frac in zip(smiles_list, fracs):
            if smiles:  # Skip empty strings
                fp = self.get_fp_bitvector(smiles, nbits)
                weighted_fp += fp * frac
        
        return weighted_fp

    def get_fps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gets a the fingerprint hashes for a set smiles strings in a dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing a column of smiles strings for which fingerprints should be generated. The column containing the fingerprints should match DoV().fingerprint_col.

        Returns:
            pd.DataFrame: Dataframe containing the hashes for the each fingerprint and a count of their occurance in the molecule.
        """

        return df.progress_apply(
            lambda row: self.get_fp(row[self.fingerprint_col]), axis=1
        ).fillna(0)

    def __count_fps_overlap(self, row, alltraincount):
        row.index = row.index.astype(int)
        testcount = row[row != 0]

        dfcount = pd.DataFrame(
            {"testcount": testcount, "alltraincount": alltraincount}
        ).fillna(0)
        dfcount = dfcount[dfcount.testcount > 0]
        returnvalue = sum(dfcount.alltraincount == 0)

        return pd.Series(returnvalue, name="min_occ")

    def get_fps_overlap(
        self, dfpredict: pd.DataFrame, dftrain_fps: pd.DataFrame
    ) -> pd.DataFrame:
        """Finds the overlaping fingerprints between the structures in the prediction dataframe and the training datafrmae.

        Args:
            dfpredict (pd.DataFrame): dataframe containing smiles for prediction. smiles should located in self.fingerprint_col
            dftrain_fps (pd.DataFrame): dataframe containing smiles that were used for training. smiles should located in self.fingerprint_col

        Returns:
            pd.DataFrame: dfpredict_fps which is the dfpredict dataframe with the count of the fingerprints outside of the training dataframe, which is located in 'fps_notin_train'.
        """
        if type(dftrain_fps) != pd.DataFrame:
            dftrain_fps = self.dftrain_fps
        alltraincount = dftrain_fps.sum(0)
        alltraincount.index = alltraincount.index.astype(int)

        dfpredict_fps = self.get_fps(dfpredict)
        dfoccur = dfpredict_fps.apply(
            lambda row: self.__count_fps_overlap(row, alltraincount), axis=1
        )
        dfoccur.columns = ["fps_notin_train"]
        return pd.concat([dfpredict, dfoccur], axis=1)

    def get_solute_solvent_overlap(
        self, 
        dfpredict: pd.DataFrame, 
        dftrain: pd.DataFrame,
        solute_col: str = None,
        solvent_col: str = None,
        solvent_frac_col: str = None,
        tanimoto_threshold: float = 0.5,
        nbits: int = 2048
    ) -> pd.DataFrame:
        """Analyzes if similar solute-solvent pairs exist in the training data.

        For each prediction sample, finds the most similar solute in the training set,
        then checks if similar solvents (or solvent mixtures) were tested with that similar solute.

        Args:
            dfpredict (pd.DataFrame): Prediction dataframe with solute and solvent columns
            dftrain (pd.DataFrame): Training dataframe with solute and solvent columns
            solute_col (str): Column name for solute SMILES (defaults to self.fingerprint_col)
            solvent_col (str): Column name for solvent SMILES (semicolon-delimited or list)
            solvent_frac_col (str): Column name for solvent mole fractions (semicolon-delimited or list)
            tanimoto_threshold (float): Minimum Tanimoto similarity to consider molecules similar
            nbits (int): Number of bits for Morgan fingerprint

        Returns:
            pd.DataFrame: Original dfpredict with additional columns:
                - max_solute_similarity: Tanimoto similarity to most similar training solute
                - max_solvent_similarity_given_solute: Max solvent similarity for the most similar solute
                - similar_pair_exists: Whether a similar solute-solvent pair exists in training
                - n_similar_solutes: Number of similar solutes found in training
                - solute_fps_notin_train: Number of solute fingerprints not in training
                - n_novel_solvents: Number of solvent components not seen in training
        """
        solute_col = solute_col or self.fingerprint_col
        solvent_col = solvent_col or self.solvent_col
        solvent_frac_col = solvent_frac_col or self.solvent_frac_col

        print("Computing solute fingerprints...")
        # Get solute fingerprints
        predict_solute_fps = dfpredict[solute_col].progress_apply(
            lambda x: self.get_fp_bitvector(x, nbits)
        )
        train_solute_fps = dftrain[solute_col].apply(
            lambda x: self.get_fp_bitvector(x, nbits)
        )

        print("Computing solvent mixture fingerprints...")
        # Get solvent mixture fingerprints
        predict_solvent_fps = []
        for _, row in tqdm(dfpredict.iterrows(), total=len(dfpredict), desc="Prediction solvents"):
            fp = self.get_mixture_fp_bitvector(
                row[solvent_col], 
                row[solvent_frac_col], 
                nbits
            )
            predict_solvent_fps.append(fp)
        
        train_solvent_fps = []
        for _, row in tqdm(dftrain.iterrows(), total=len(dftrain), desc="Training solvents"):
            fp = self.get_mixture_fp_bitvector(
                row[solvent_col], 
                row[solvent_frac_col], 
                nbits
            )
            train_solvent_fps.append(fp)

        results = []
        
        print("Analyzing solute-solvent pair similarities...")
        for idx, (pred_solute_fp, pred_solvent_fp) in tqdm(
            enumerate(zip(predict_solute_fps, predict_solvent_fps)),
            total=len(dfpredict),
            desc="Processing predictions"
        ):
            # Find most similar solute in training set
            solute_similarities = [
                DataStructs.TanimotoSimilarity(
                    DataStructs.ExplicitBitVect(nbits, pred_solute_fp.tolist()),
                    DataStructs.ExplicitBitVect(nbits, train_fp.tolist())
                )
                for train_fp in train_solute_fps
            ]
            max_solute_sim = max(solute_similarities) if solute_similarities else 0.0
            
            # Find indices of similar solutes (above threshold)
            similar_solute_indices = [
                i for i, sim in enumerate(solute_similarities) 
                if sim >= tanimoto_threshold
            ]
            
            # For the most similar solute(s), check solvent similarity
            # Use cosine similarity for weighted fingerprints
            max_solvent_sim = 0.0
            if similar_solute_indices:
                for train_idx in similar_solute_indices:
                    train_solvent_fp = train_solvent_fps[train_idx]
                    # Cosine similarity for weighted fingerprints
                    dot_product = np.dot(pred_solvent_fp, train_solvent_fp)
                    norm_pred = np.linalg.norm(pred_solvent_fp)
                    norm_train = np.linalg.norm(train_solvent_fp)
                    if norm_pred > 0 and norm_train > 0:
                        solvent_sim = dot_product / (norm_pred * norm_train)
                    else:
                        solvent_sim = 0.0
                    max_solvent_sim = max(max_solvent_sim, solvent_sim)
            
            # Check if similar pair exists
            similar_pair_exists = (
                max_solute_sim >= tanimoto_threshold and 
                max_solvent_sim >= tanimoto_threshold
            )
            
            results.append({
                'max_solute_similarity': max_solute_sim,
                'max_solvent_similarity_given_solute': max_solvent_sim,
                'similar_pair_exists': similar_pair_exists,
                'n_similar_solutes': len(similar_solute_indices)
            })
        
        # Add fingerprint overlap counts
        print("Computing fingerprint overlaps...")
        old_fingerprint_col = self.fingerprint_col
        
        # Solute overlap
        self.fingerprint_col = solute_col
        solute_fps_train = self.get_fps(dftrain)
        dfpredict_with_solute = self.get_fps_overlap(dfpredict, solute_fps_train)
        
        # For solvent mixtures, compute overlap for each component
        print("Computing solvent component overlaps...")
        solvent_overlaps = []
        for _, pred_row in tqdm(dfpredict.iterrows(), total=len(dfpredict), desc="Solvent overlaps"):
            pred_solvents = self._parse_solvent_list(pred_row[solvent_col])
            
            # Collect all training solvents
            train_solvents = set()
            for _, train_row in dftrain.iterrows():
                train_solv = self._parse_solvent_list(train_row[solvent_col])
                train_solvents.update(train_solv)
            
            # Count how many prediction solvents are not in training
            n_novel = sum(1 for s in pred_solvents if s not in train_solvents)
            solvent_overlaps.append(n_novel)
        
        # Restore original fingerprint column
        self.fingerprint_col = old_fingerprint_col
        
        # Combine results
        df_results = pd.DataFrame(results)
        df_results['solute_fps_notin_train'] = dfpredict_with_solute['fps_notin_train'].values
        df_results['n_novel_solvents'] = solvent_overlaps
        
        return pd.concat([dfpredict.reset_index(drop=True), df_results], axis=1)

    def get_nearest_neighbor_analysis(
        self,
        dfpredict: pd.DataFrame,
        dftrain: pd.DataFrame,
        solute_col: str = None,
        solvent_col: str = None,
        solvent_frac_col: str = None,
        k: int = 5,
        nbits: int = 2048
    ) -> pd.DataFrame:
        """Find k-nearest neighbors in the training set for each prediction sample.

        Args:
            dfpredict (pd.DataFrame): Prediction dataframe
            dftrain (pd.DataFrame): Training dataframe
            solute_col (str): Column name for solute SMILES
            solvent_col (str): Column name for solvent SMILES (semicolon-delimited or list)
            solvent_frac_col (str): Column name for solvent mole fractions (semicolon-delimited or list)
            k (int): Number of nearest neighbors to find
            nbits (int): Number of bits for Morgan fingerprint

        Returns:
            pd.DataFrame: Original dfpredict with columns for each k-NN similarity
        """
        solute_col = solute_col or self.fingerprint_col
        solvent_col = solvent_col or self.solvent_col
        solvent_frac_col = solvent_frac_col or self.solvent_frac_col

        print("Computing combined solute-solvent fingerprints...")
        
        # Combine solute and solvent mixture fingerprints
        predict_combined_fps = []
        for _, row in tqdm(dfpredict.iterrows(), total=len(dfpredict), desc="Prediction FPs"):
            solute_fp = self.get_fp_bitvector(row[solute_col], nbits)
            solvent_fp = self.get_mixture_fp_bitvector(
                row[solvent_col], 
                row[solvent_frac_col], 
                nbits
            )
            combined_fp = np.concatenate([solute_fp, solvent_fp])
            predict_combined_fps.append(combined_fp)
        
        train_combined_fps = []
        for _, row in tqdm(dftrain.iterrows(), total=len(dftrain), desc="Training FPs"):
            solute_fp = self.get_fp_bitvector(row[solute_col], nbits)
            solvent_fp = self.get_mixture_fp_bitvector(
                row[solvent_col], 
                row[solvent_frac_col], 
                nbits
            )
            combined_fp = np.concatenate([solute_fp, solvent_fp])
            train_combined_fps.append(combined_fp)
        
        print(f"Finding {k}-nearest neighbors...")
        knn_results = []
        
        for pred_fp in tqdm(predict_combined_fps, desc="Computing similarities"):
            similarities = []
            for train_fp in train_combined_fps:
                # Cosine similarity for weighted fingerprints
                dot_product = np.dot(pred_fp, train_fp)
                norm_pred = np.linalg.norm(pred_fp)
                norm_train = np.linalg.norm(train_fp)
                if norm_pred > 0 and norm_train > 0:
                    sim = dot_product / (norm_pred * norm_train)
                else:
                    sim = 0.0
                similarities.append(sim)
            
            # Get k largest similarities
            top_k_sims = sorted(similarities, reverse=True)[:k]
            
            result = {f'knn_{i+1}_similarity': sim for i, sim in enumerate(top_k_sims)}
            result['mean_knn_similarity'] = np.mean(top_k_sims)
            result['min_knn_similarity'] = np.min(top_k_sims)
            knn_results.append(result)
        
        df_knn = pd.DataFrame(knn_results)
        return pd.concat([dfpredict.reset_index(drop=True), df_knn], axis=1)

    @property
    def radius(self):
        """The radius that will be used for Morgan fingerprinting."""
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius

    @property
    def fingerprint_col(self):
        """The column which should contain canonnical smiles for which fingerprint hashes will be generated."""
        return self._fingerprint_col

    @fingerprint_col.setter
    def fingerprint_col(self, fingerprint_col):
        self._fingerprint_col = fingerprint_col

    @property
    def solvent_col(self):
        """The column which should contain solvent SMILES (semicolon-delimited or list)."""
        return self._solvent_col

    @solvent_col.setter
    def solvent_col(self, solvent_col):
        self._solvent_col = solvent_col

    @property
    def solvent_frac_col(self):
        """The column which should contain solvent mole fractions (semicolon-delimited or list)."""
        return self._solvent_frac_col

    @solvent_frac_col.setter
    def solvent_frac_col(self, solvent_frac_col):
        self._solvent_frac_col = solvent_frac_col