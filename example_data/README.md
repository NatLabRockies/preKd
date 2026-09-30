# Example data

`example_measurements.csv` is a **synthetic** 40-row table (10 solutes x 4 solvent
systems) provided so the pipeline can be run end to end without any real data.

The `log_kp` values are a smooth deterministic function of each solute's computed
logP and TPSA — they are **not measurements** and carry no physical meaning. Use
them to check that installation, data preparation, training and prediction work;
do not use them to judge accuracy or to train a model you intend to trust.

It does illustrate the expected input format:

| column | contents |
| --- | --- |
| `solute_smiles` | SMILES of the solute |
| `solvent_smiles` | `;`-delimited SMILES of the solvent-system components |
| `solvent_mol_fractions` | `;`-delimited mole fractions, matching the solvent order and summing to 1 |
| `log_kp` | the target value |

Any extra columns (here `compound_name` and `solvent_system`) are carried through
untouched and appear again in the prediction output.
