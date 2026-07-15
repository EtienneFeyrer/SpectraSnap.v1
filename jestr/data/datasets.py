import matchms
from matchms.importing import load_from_json
from matchms import Spectrum
import pandas as pd
import json
import typing as T
import numpy as np
import torch
import massspecgym.utils as utils
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
import dgl
from collections import defaultdict
from massspecgym.data.transforms import SpecTransform, MolTransform, MolToInChIKey
from massspecgym.data.datasets import MassSpecDataset, UnlabeledDataset, MassDataset
import jestr.utils.data as data_utils
from torch.nn.utils.rnn import pad_sequence
from massspecgym.models.base import Stage
import pickle
import math
import itertools
from rdkit.Chem import AllChem
from rdkit import Chem
class JESTR1_MassSpecDataset(MassSpecDataset):
    def __init__(
        self,
        spectra_view: str,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.spectra_view = spectra_view

    def __getitem__(self, i, transform_spec: bool = True, transform_mol: bool = True):

        spec = self.spectra[i]
        metadata = self.metadata.iloc[i]
        mol = metadata["smiles"]

        # Apply all transformations to the spectrum
        item = {}
        if transform_spec and self.spec_transform:
            if isinstance(self.spec_transform, dict):
                for key, transform in self.spec_transform.items():
                    item[key] = transform(spec) if transform is not None else spec
            else:
                item["spec"] = self.spec_transform(spec)
        else:
            item["spec"] = spec

        if self.return_mol_freq:
            item["mol_freq"] = metadata["mol_freq"]

        if self.return_identifier:
            item["identifier"] = metadata["identifier"]
            
        # Apply all transformations to the molecule
        if transform_mol and self.mol_transform:
            if isinstance(self.mol_transform, dict):
                for key, transform in self.mol_transform.items():
                    item[key] = transform(mol) if transform is not None else mol
            else:
                item["mol"] = self.mol_transform(mol)
        else:
            item["mol"] = mol
        return item

class ContrastiveDataset(Dataset):
    def __init__(
        self,
        spec_mol_data,
    ):
        super().__init__()
    
        indices = spec_mol_data.indices
        self.spec_mol_data = spec_mol_data
        self.smiles_to_specmol_ids = spec_mol_data.dataset.metadata.loc[indices].groupby('smiles').indices
        self.smiles_to_spec_couter = defaultdict(int)
        self.smiles_list = list(self.smiles_to_specmol_ids.keys())

    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, i:int) -> dict:
        mol = self.smiles_list[i]

        # select spectrum (iterate through list of spectra)
        specmol_ids = self.smiles_to_specmol_ids[mol]
        counter = self.smiles_to_spec_couter[mol]
        specmol_id = specmol_ids[counter % len(specmol_ids)]

        item = self.spec_mol_data.__getitem__(specmol_id)
        self.smiles_to_spec_couter[mol] = counter+1
        # item['smiles'] = mol
        # item['spec_id'] = specmol_id
        return item

    @staticmethod
    def collate_fn(batch: T.List[dict], spec_enc: str, spectra_view: str, stage=None) -> dict:
        stage = Stage(stage)
        if stage == Stage.ONLINE_TEST:
            spec_key = spectra_view[0] if isinstance(spectra_view, (list, tuple)) else spectra_view
            if spec_key in batch[0]:
                item_spec_key = spec_key
            elif "spec" in batch[0]:
                item_spec_key = "spec"
            else:
                raise KeyError(
                    f"Expected spectrum key '{spec_key}' or 'spec' in batch item; got keys: {list(batch[0].keys())}"
                )

            collated_batch = {}
            # Stack spectra and collect IDs for downstream FAISS lookup.
            specs = [item[item_spec_key] for item in batch]
            spec_ids = [item["spec_id"] for item in batch]
            #neutral_masses = [item["neutral_mass"] for item in batch]
            precursor_mzs = [item["precursor_mz"] for item in batch]
            spec_identifiers = [item.get("identifier") for item in batch]
            
            collated_batch[item_spec_key] = torch.stack(specs) if isinstance(specs[0], torch.Tensor) else default_collate(specs)
            collated_batch["spec_id"] = spec_ids
            #collated_batch["neutral_mass"] = neutral_masses
            collated_batch["precursor_mz"] = precursor_mzs
            collated_batch["identifier"] = spec_identifiers
            
        elif stage == Stage.PRECOMPUTE:
            #print(f"DEBUG: collate_fn called with batch size {len(batch)}, stage=PRECOMPUTE")
            # Each batch item is one group of candidate molecules.
            cand_groups = [item['cand'] for item in batch]
            cand_smiles_groups = [item.get('cand_smiles', []) for item in batch]
            group_ids = [item.get('group_id') for item in batch if 'group_id' in item]
            group_sizes = [len(group) for group in cand_groups]
           # print(f"DEBUG: cand_groups sizes: {group_sizes}")

            # Flatten graphs to feed throught encoder, with group boundaries for re-splitting.
            flat_graphs = [mol for group in cand_groups for mol in group]
           # print(f"DEBUG: flattened to {len(flat_graphs)} graphs")
            collated_batch = {
                'cand_smiles_groups': cand_smiles_groups,
                'group_sizes': group_sizes,
            }
            if len(group_ids) == len(batch):
                collated_batch['group_id'] = group_ids
            if len(flat_graphs) > 0:
                #print(f"DEBUG: batching {len(flat_graphs)} DGL graphs")
                collated_batch['cand'] = dgl.batch(flat_graphs)
                #print(f"DEBUG: Successfully batched graphs")
            else:
                #print(f"DEBUG: No graphs to batch")
                collated_batch['cand'] = None
        else:
            mol_key = 'cand' if stage == Stage.TEST else 'mol'
            non_standard_collate = ['mol', 'cand']

            collated_batch = {}
            # standard collate
            for k in batch[0].keys():
                if k not in non_standard_collate:
                    collated_batch[k] = default_collate([item[k] for item in batch])

            # batch graphs
            batch_mol = []
            batch_mol_nodes = []
            for item in batch:
                batch_mol.append(item[mol_key])
                batch_mol_nodes.append(item[mol_key].num_nodes())

            collated_batch[mol_key] = dgl.batch(batch_mol)
            collated_batch['mol_n_nodes'] = batch_mol_nodes
        collated_batch['stage'] = stage
        return collated_batch

class ExpandedRetrievalDataset:
    '''Used for testing only 
    Assumes 'fold' column defines the split'''
    def __init__(self,
                 mol_label_transform: MolTransform = MolToInChIKey(),
                 candidates_pth: T.Optional[T.Union[Path, str]] = None,
                 candidate_file_key: str = "smiles",
                **kwargs):
        
        self.instance = JESTR1_MassSpecDataset(**kwargs, return_mol_freq=False)
        # super().__init__(**kwargs)

        self.candidates_pth = candidates_pth
        self.mol_label_transform = mol_label_transform
        
        #resolve/download candidates file if not provided similar to RetrievalDataset from massspecgym
        if self.candidates_pth is None:
            # default to the formula candidates file (same as RetrievalDataset)
            print("No path provided downloads from hugging face and stores file in .cache/huggingface/datasets/molecules/MassSpecGym_retrieval_candidates_formula.json")
            self.candidates_pth = utils.hugging_face_download(
                "molecules/MassSpecGym_retrieval_candidates_formula.json"
            )
        elif isinstance(self.candidates_pth, str):
            if Path(self.candidates_pth).is_file():
                self.candidates_pth = Path(self.candidates_pth)
            else:
                # allow passing a relative hf path string to download
                self.candidates_pth = utils.hugging_face_download(self.candidates_pth)

        # Read candidates_pth from json to dict: SMILES -> respective candidate SMILES
        with open(self.candidates_pth, "r") as file:
            candidates = json.load(file)
        self.candidates = {}
        for s, cand in candidates.items():
        #TODO: check if mol to graph is true from main
            self.candidates[s] = [c for c in cand if '.' not in c]
        
        self.spec_cand = [] #(spec index, cand_smiles, true_label)
        
        
        if 'smiles' not in self.metadata.columns or candidate_file_key == "identifier":
            print("WARNING: 'smiles' column not found in metadata; using 'identifier' column for candidate matching")
            if not isinstance(self.metadata.iloc[0]['identifier'], str):
                self.metadata['smiles'] = self.metadata['identifier'].apply(str)
            else:
                self.metadata['smiles'] = self.metadata['identifier']

        test_smiles = self.metadata[self.metadata['fold'] == "test"]['smiles'].tolist()
        test_ms_id = self.metadata[self.metadata['fold'] == "test"]['identifier'].tolist()
        
        # only keep the unique first occuring spectra for each smiles
        appeared_smiles = set()
        unique_test_smiles = []
        unique_test_ms_id = []
        for smiles, spec_id in zip(test_smiles, test_ms_id):
            if smiles not in appeared_smiles:
                appeared_smiles.add(smiles)
                unique_test_smiles.append(smiles)
                unique_test_ms_id.append(spec_id)

       # print(f"print first test smiles:{unique_test_ms_id[:3]} and corresponding ms ids: {unique_test_ms_id[:5]}")
        # use the metadata index for later retrieval of spectrum metadata by index
        spec_id_to_index = dict(zip(self.metadata['identifier'], self.metadata.index))
        for spec_id, s in zip(unique_test_ms_id, unique_test_smiles):
           # print(f"DEBUG target spec_id={spec_id} target={s!r} type={type(s).__name__} in_candidates={s in self.candidates}")
            candidates = self.candidates[s]
            candidates = [c for c in candidates if c != s]
            # mol_label = self.mol_label_transform(s)
            # labels = [self.mol_label_transform(c) == mol_label for c in candidates]
            labels = [c == s for c in candidates]
            if len(candidates) == 0:
                print(f"Skipping {spec_id}; empty candidate set")
                continue
            # if not any(labels):
            #     print(f"Target smiles not in candidate set")

            self.spec_cand.extend([(spec_id_to_index[spec_id], s, True)] + [(spec_id_to_index[spec_id], candidates[j], k) for j, k in enumerate(labels)])
    
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)
    
    def __len__(self):
        return len(self.spec_cand)

    def __getitem__(self, i):
        spec_i = self.spec_cand[i][0]
        cand_smiles = self.spec_cand[i][1]
        label = self.spec_cand[i][2]

        item = self.instance.__getitem__(spec_i, transform_mol=False)
        item['cand'] = self.mol_transform(cand_smiles)
        item['cand_smiles'] = cand_smiles
        item['label'] = label
        return item

# class ExpandedRetrievalDataset:
#     '''Used for testing only 
#     Assumes 'fold' column defines the split'''
#     def __init__(self,
#                  mol_label_transform: MolTransform = MolToInChIKey(),
#                  candidates_pth: T.Optional[T.Union[Path, str]] = None,
#                  candidate_file_key: str = "smiles",
#                 **kwargs):
        
#         self.instance = JESTR1_MassSpecDataset(**kwargs, return_mol_freq=False)
#         # super().__init__(**kwargs)

#         print("Parent calss metadata columns:", self.metadata.columns.tolist())

#         self.candidates_pth = candidates_pth
#         self.mol_label_transform = mol_label_transform
        
#         #resolve/download candidates file if not provided similar to RetrievalDataset from massspecgym
#         if self.candidates_pth is None:
#             # default to the formula candidates file (same as RetrievalDataset)
#             print("No path provided downloads from hugging face and stores file in .cache/huggingface/datasets/molecules/MassSpecGym_retrieval_candidates_formula.json")
#             self.candidates_pth = utils.hugging_face_download(
#                 "molecules/MassSpecGym_retrieval_candidates_formula.json"
#             )
#         elif isinstance(self.candidates_pth, str):
#             if Path(self.candidates_pth).is_file():
#                 self.candidates_pth = Path(self.candidates_pth)
#             else:
#                 # allow passing a relative hf path string to download
#                 self.candidates_pth = utils.hugging_face_download(self.candidates_pth)

#         # Read candidates_pth from json to dict: SMILES -> respective candidate SMILES
#         with open(self.candidates_pth, "r") as file:
#             candidates = json.load(file)

#         self.candidates = {}
#         # s is the key here smiles
#         for s, cand in candidates.items():
#             self.candidates[s] = [c for c in cand if '.' not in c]
        
#         self.spec_cand = [] #(spec index, cand_smiles, true_label)

#         # this checks if we have to work with identifiers
#         if 'smiles' not in self.metadata.columns or candidate_file_key == "identifierss":
#             if not isinstance(self.metadata.iloc[0]['identifier'], str):
#                 self.metadata['smiles'] = self.metadata['identifier'].apply(str)
#             else:
#                 self.metadata['smiles'] = self.metadata['identifier']
        
        
#         test_smiles = self.metadata[self.metadata['fold'] == "test"]['smiles'].tolist()
#         test_ms_id = self.metadata[self.metadata['fold'] == "test"]['identifier'].tolist()
#         # only keep test first occuring spectra for each smiles (to avoid duplicates in test set)
#         appeared_smiles = set()
#         unique_test_smiles = []
#         unique_test_ms_id = []
#         for smiles, spec_id in zip(test_smiles, test_ms_id):
#             if smiles not in appeared_smiles:
#                 appeared_smiles.add(smiles)
#                 unique_test_smiles.append(smiles)
#                 unique_test_ms_id.append(spec_id)


#         spec_id_to_index = dict(zip(self.metadata['identifier'], self.metadata.index))
#         for spec_id, s in zip(unique_test_ms_id, unique_test_smiles):
#             candidates = self.candidates[s]
#             # mol_label = self.mol_label_transform(s)
#             # labels = [self.mol_label_transform(c) == mol_label for c in candidates]
#             #remove target
#             candidates = [c for c in candidates if c != s]

#             labels = [c == s for c in candidates]
#             if len(candidates) == 0:
#                 print(f"Skipping {spec_id}; empty candidate set")
#                 continue
#             # if not any(labels):
#             #     print(f"Target smiles not in candidate set")

#             #add target as first candidate with label True
#             new_entry = [(spec_id_to_index[spec_id], s, True)]
#             new_entry.extend([(spec_id_to_index[spec_id], candidates[j], k) for j, k in enumerate(labels)])
            
#             # spectrum index with candidates with labels
#             self.spec_cand.extend(new_entry)
    
#     def __getattr__(self, name):
#         #accesing any attribute not found from parent class
#         return self.instance.__getattribute__(name)
    
#     def __len__(self):
#         return len(self.spec_cand)

#     def __getitem__(self, i):
#         spec_i = self.spec_cand[i][0]
#         cand_smiles = self.spec_cand[i][1]
#         label = self.spec_cand[i][2]

#         item = self.instance.__getitem__(spec_i, transform_mol=False)
#         item['cand'] = self.mol_transform(cand_smiles)
#         item['cand_smiles'] = cand_smiles
#         item['label'] = label
#         return item

 
#TODO: Datasets for precomputation and online testing 
class PrecomputeCandDataset(UnlabeledDataset):
    """Dataset for precomputing candidate embeddings.
       Inherits from UnlabeledDataset.
    - Keeps raw candidate list groups from the base UnlabeledDataset.
    - Removes candidates containing '.' (dot-containing entries are always filtered dots lead to error in mol_transform).
    - Optionally applies mol_transform to each candidate in a group.
    """
    def __init__(self, mol_transform: T.Optional[MolTransform] = None, mol_view: T.Optional[T.Union[str, T.List[str]]] = None, **kwargs):
        # kwargs are passed to the base UnlabeledDataset (expects candidates_pth, keys_order, ...)
        super().__init__(**kwargs, expected_type=list)
        self.mol_transform = mol_transform
        self.mol_view = mol_view

        # UnlabeledDataset stores the loaded JSON in self.data (list-of-lists expected)
        # Keep list-of-lists form so each dataset idx corresponds to one candidate group.
        # Filter out entries containing '.' and remove any groups that become empty
        self.data = [
            [c for c in cand_list if "." not in c]
            for cand_list in self.data
        ]
        # Remove empty candidate groups to avoid producing None batches during collate
        orig_len = len(self.data)
        self.data = [group for group in self.data if len(group) > 0]
        if len(self.data) < orig_len:
            print(f"WARNING: Removed {orig_len - len(self.data)} empty candidate groups during PrecomputeCandDataset initialization")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        cand_smiles_group = self.data[i]

        # Transform each SMILES into a DGLGraph
        if self.mol_transform is not None:
            transformed = []
            for c in cand_smiles_group:
                try:
                    graph = self.mol_transform(c)
                    transformed.append(graph)
                except Exception as e:
                    print(f"ERROR: Failed to transform SMILES '{c}': {type(e).__name__}: {e}")
                    raise
        else:
            return {"cand_smiles": cand_smiles_group, "cand": None, "group_id": i}
            # raise ValueError("mol_transform must be provided for PrecomputeCandDataset")

       # print(f"DEBUG: Transformed {len(cand_smiles_group)} SMILES to {len(transformed)} graphs")
        return {
            "cand": transformed,              # list of DGLGraphs
            "cand_smiles": cand_smiles_group, # list of SMILES
            "group_id": i
        }

class PrecomputeBinsDataset(MassDataset):
    """
    Dataset to precompute embedding bins and FAISS index.
    Each item corresponds to one mass bin.

    Output format:
        {
            "cand": List[DGLGraph] or None,
            "cand_smiles": List[str],
            "group_id": mass_bin_key
        }
    """

    def __init__(
        self,
        mol_transform: T.Optional[MolTransform] = None,
        mol_view: T.Optional[T.Union[str, T.List[str]]] = None,
        **kwargs
    ):
        # MassDataset loads a dict: {mass_bin: [smiles]}
        super().__init__(**kwargs)

        self.mol_transform = mol_transform
        self.mol_view = mol_view

        # Filter out SMILES containing '.' for every mass bin
        filtered = {}
        removed_groups = 0

        for mass, smiles_list in self.data.items():
            clean = [s for s in smiles_list if "." not in s]
            if len(clean) == 0:
                removed_groups += 1
                continue
            filtered[mass] = clean

        if removed_groups > 0:
            print(f"WARNING: Removed {removed_groups} empty or invalid mass bins during PrecomputeBinsDataset initialization")

        # Replace internal data + keys
        self.data = filtered
        self.keys = list(filtered.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        mass_bin = self.keys[idx]
        cand_smiles = self.data[mass_bin]

        # If no transform, return SMILES only
        if self.mol_transform is None:
            return {
                "cand": None,
                "cand_smiles": cand_smiles,
                "group_id": mass_bin
            }

        # Apply mol_transform to each SMILES
        transformed = []
        for sm in cand_smiles:
            try:
                g = self.mol_transform(sm)
                transformed.append(g)
            except Exception as e:
                print(f"ERROR: Failed to transform SMILES '{sm}': {type(e).__name__}: {e}")
                raise

        return {
            "cand": transformed,      # list of DGLGraphs
            "cand_smiles": cand_smiles,
            "group_id": mass_bin      # mass bin centroid
        }

class SpecDataset(UnlabeledDataset):
    """Dataset for online spectrum embedding computation 
    or TODO: Wrapper Method in training.
       Inherits from UnlabeledDataset.
    - Builds a list of spectra items from the base UnlabeledDataset.
    - Spectrum items are dicts containing all neccessary information.
    - Applies spec_transform per-item when returned.
    """
    def __init__(self, spec_transform: T.Optional[SpecTransform] = None, **kwargs):
        # kwargs are passed to the base UnlabeledDataset (expects dataset_pth, keys_order, ...)
        super().__init__(raw_pth=kwargs.get('spectra_pth'), expected_type=matchms.Spectrum)
        self.spec_transform = spec_transform
        # UnlabeledDataset stores the loaded list directly in self.data
        self.spectra = self.data

    def json_load(self, pth: Path) -> T.List[matchms.Spectrum]:
        return list(load_from_json(pth))

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, i, transform_spec: bool = True):
        spec = self.spectra[i]
        item = {}
        if transform_spec and self.spec_transform:
            if isinstance(self.spec_transform, dict):
                for key, transform in self.spec_transform.items():
                    item[key] = transform(spec) if transform is not None else spec
            else:
                item["spec"] = self.spec_transform(spec)
        else:
            item["spec"] = spec
        
        # metadata not available for plain unlabeled spectra; return index as identifier
        #TODO: should include the neutral mass for later reference to bins
        item["spec_id"] = i
        #print(f"DEBUG: Retrieved spectrum with metadata: {spec.metadata}")
        #item["neutral_mass"] = spec.get("neutral_mass")
        item["precursor_mz"] = spec.get("precursor_mz")
        item["identifier"] = spec.get("identifier")
        return item
        
        # if self.spec_transform is not None:
        #     transformed = self.spec_transform(spec)
        #     # convert numpy arrays to torch tensors for convenience
        #     if isinstance(transformed, np.ndarray):
        #         transformed = torch.as_tensor(transformed)
        # else:
        #     transformed = spec
        # # metadata not available for plain unlabeled spectra; return index as identifier
        # return {"spec": transformed, "spec_id": i}


class CandDataset(UnlabeledDataset):
    """Dataset for unlabelled candidates for a Wrapper Method
    that produces pseudo labels for contrastive learning."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs, expected_type=str)
    # TODO: Wrapper Method in training.