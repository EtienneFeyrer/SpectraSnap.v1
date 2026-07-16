import argparse
import datetime
import sys
import os
import json
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jestr.data.datasets as jestr_datasets
import pandas as pd

from rdkit import RDLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from massspecgym.models.base import Stage

from jestr.data.data_module import PredictDataModule
from jestr.data.datasets import ContrastiveDataset
from massspecgym.data.datasets import UnlabeledDataset
from jestr.utils.data import get_spec_featurizer, get_mol_featurizer, get_test_ms_dataset
from jestr.utils.models import get_model

from jestr.definitions import EXPERIMENTS_DIR
import yaml
from functools import partial
# Suppress RDKit warnings and errors
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("--param_pth", type=str, default="params_pub_faiss_inference.yaml")
parser.add_argument('--checkpoint_pth', type=str, default='')
parser.add_argument('--checkpoint_choice', type=str, default='train', choices=['train', 'val'])
parser.add_argument('--df_test_pth', type=str, help='result file name')
parser.add_argument('--candidates_pth', type=str)
parser.add_argument('--faiss_index_dir', type=str, help='faiss index directory')
parser.add_argument('--exp_dir', type=str)
parser.add_argument('--k', type=int, default=20, help='number of candidates to retrieve with faiss')
parser.add_argument('--spectra_pth', type=str, help='path to spectra file for inference')

# Inference has to load the unlabeled spectra with their identifiers 
# then the spectra will be encoded and for each a faiss search will be 
# performed to retrieve the top k candidates

# has to be changed to return a smiles list based on the index returned
def faiss_to_unified_format(faiss_results, dict_of_candidates):
    unified_results = []
    flat_dict = {}
    flat_dict = {}
    collisions = []

    for batch_output in faiss_results:
        for key, value in batch_output.items():
            if key in flat_dict:
                collisions.append(key)
            flat_dict[key] = value

    #print(f"DEBUG: size of dictionary with all faiss results: {len(flat_dict)}")
    #print(f"DEBUG: number of collisions: {len(collisions)}")
    #print(f"DEBUG: collisions: {collisions}")
    # iterate in the same order as input
    for i, (identifier, entry) in enumerate(flat_dict.items()):
        results = entry["results"]

        candidate_smiles = []
        scores = []
        ppm_errors = []

        for cand_idx, cand_info in results.items():
            if cand_idx == -1:
                continue  # skip invalid FAISS entries
            bin_idx = str(entry["bin"])
            candidate_smiles.append(dict_of_candidates.data[bin_idx][cand_idx])
            scores.append(cand_info["distance"])
            ppm_errors.append(cand_info["ppm_error"])

        unified_results.append({
            "identifier": identifier,
            "embedding": entry["embeddings"],
            "candidates": candidate_smiles,
            "scores": scores,
            "ppm_error": ppm_errors,
            "bin": bin_idx
        })

    return pd.DataFrame(unified_results)

def main(params):
    # Seed everything
    pl.seed_everything(params['seed'])

    # Load dataset
    spec_featurizer = get_spec_featurizer(params['spectra_view'], params)
    #mol_featurizer = get_mol_featurizer(params['molecule_view'], params)
    #Assign precompute dataset
    dataset = get_test_ms_dataset(spectra_view=params['spectra_view'], spectra_featurizer=spec_featurizer, params=params)

    # Init data module
    collate_fn = partial(ContrastiveDataset.collate_fn, spec_enc=params['spec_enc'], spectra_view=params['spectra_view'], stage=params['stage'])

    data_module = PredictDataModule(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=1, # Must be 1: each spectrum has its own index file
        num_workers=1
    )

    model = get_model(params=params, stage= params['stage'], model =params['model'],)
    model.faiss_index_path = params['faiss_index_dir']
    model.k = params['k']

    # Init trainer
    trainer = Trainer(
        accelerator=params['accelerator'],
        devices=params['devices'],
        default_root_dir=params['experiment_dir']
    )

    # Prepare data module to test
    data_module.prepare_data()
    data_module.setup(stage="predict")
        
    # Test the computed faiss indexes return indices
    outputs = trainer.predict(model, datamodule=data_module)
    #trainer.test(model, datamodule=data_module)
    print(f"Finished inference")
    print(f"outputs length: {len(outputs)}")
    #save test results to disk
    output_path = os.path.join(params['experiment_dir'], 'outputs.json')
    with open(output_path, 'w') as f:
        json.dump(outputs, f)
    print(f"Finished testing. Saving results to disk path: {output_path}")

    #return
    
    #create ublabeled datasets for mapping back faiss results to identifiers and candidates
    unlabeled_spectrum = jestr_datasets.SpecDataset(spectra_pth=params['spectra_pth'], spec_transform=spec_featurizer)
    dict_of_candidates = jestr_datasets.PrecomputeBinsDataset(raw_pth=params.get('candidates_pth'))

    unified_format = faiss_to_unified_format(outputs, dict_of_candidates)
    unified_format.to_pickle(params['df_test_path'])
    print(f"Finished converting faiss results to unified format. Saving to disk path: {params['df_test_path']}")

    # compute recall metrics
    # recall1 = 0
    # recall5 = 0
    # recall20 = 0
    # for spectrum_output in outputs:
    #     # ID rank list list of the canddiates the 0 is the target molecule
    #     ID_rank_list =list(spectrum_output.keys())[:20]
    #     if 0 in ID_rank_list[:1]:
    #         recall1 += 1
    #     if 0 in ID_rank_list[:5]:
    #         recall5 += 1
    #     if 0 in ID_rank_list[:20]:
    #         recall20 += 1
    # recall1 /= len(outputs)
    # recall5 /= len(outputs)
    # recall20 /= len(outputs)
    # # save recall metrics to disk
    # recall_metrics = {
    #     'recall@1': recall1,
    #     'recall@5': recall5,
    #     'recall@20': recall20   }
    # recall_path = os.path.join(params['experiment_dir'], 'recall_metrics.json')
    # with open(recall_path, 'w') as f:        json.dump(recall_metrics, f)
    # print(f"Finished computing recall metrics. Saving to disk path: {recall_path}")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Load
    with open(args.param_pth) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    # Experiment directory
    exp_dir = ''
    if args.exp_dir:
        exp_dir = args.exp_dir
    else:
        run_name = params['run_name']
        for exp in os.listdir(EXPERIMENTS_DIR): # find exp dir with matching run_name
            if exp.endswith("_"+run_name):
                exp_dir = str(EXPERIMENTS_DIR / exp)
                break
    if not exp_dir: # if no exp dir provided, create a new one
        now = datetime.datetime.now().strftime("%Y%m%d")
        exp_dir = str(EXPERIMENTS_DIR / f"{now}_{params['run_name']}")
        os.makedirs(exp_dir, exist_ok=True)
    print("EXPERIMENT directory: ",exp_dir)
    params['experiment_dir'] = exp_dir
    
    # Look for checkpoint
    if args.checkpoint_pth:
        params['checkpoint_pth'] = args.checkpoint_pth
    elif params['checkpoint_pth'] != '':
        pass
    elif params['checkpoint_pth_spec_enc'] != '':
        pass
    else:
        print("No checkpoint provided. Using the checkpoint in the experiment directory")
        for f in os.listdir(exp_dir):
            if f.endswith("ckpt") and f.startswith("epoch") and args.checkpoint_choice in f:
                checkpoint_path = os.path.join(exp_dir, f)
                params['checkpoint_pth'] = checkpoint_path
                break
        if not params['checkpoint_pth']:
            raise ValueError("No checkpoint provided. Please provide a checkpoint path.")
    #only need spectra and faiss index for inference
    # also need candidates path for mapping faiss indizes back to 
    # molecules
    if args.faiss_index_dir:
        params['faiss_index_dir'] = args.faiss_index_dir
    if args.spectra_pth:
        params['spectra_pth'] = args.spectra_pth
        print("Spectra path for inference: ", params['spectra_pth'])
    if args.candidates_pth:
        params['candidates_pth'] = args.candidates_pth
    if args.k:
        params['k'] = args.k
    #if not params['df_test_path']:
        params['df_test_path'] = os.path.join(exp_dir, f"result_{params['spectra_pth'].split('/')[-1].split('.')[0]}.pkl")
        print("Test result file path: ", params['df_test_path'])
    else:
        TypeError("No spectra path provided. Please provide a spectra path for inference.")
    
    start_time = time.time()
    main(params)
    end_time = time.time()
    print(f"Total inference time: {end_time - start_time:.2f} seconds)")


