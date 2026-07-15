import argparse
import datetime
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rdkit import RDLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from massspecgym.models.base import Stage
import json
import numpy as np

from jestr.data.data_module import PredictDataModule
from jestr.data.datasets import ContrastiveDataset
from jestr.utils.data import get_mol_featurizer, get_test_ms_dataset
from jestr.utils.models import get_model
import faiss
from jestr.definitions import EXPERIMENTS_DIR
import yaml
from functools import partial
# Suppress RDKit warnings and errors
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("--param_pth", type=str, default="params_pub_faiss_precompute.yaml")
parser.add_argument('--checkpoint_pth', type=str, default='')
parser.add_argument('--checkpoint_choice', type=str, default='train', choices=['train', 'val'])
parser.add_argument('--candidates_pth', type=str)
parser.add_argument('--faiss_index_dir', type=str, help='faiss index directory')
parser.add_argument('--exp_dir', type=str)

def main(params):

    # TODO: uncomment 

    # Seed everything
    # pl.seed_everything(params['seed'])

    # # Load dataset
    # #spec_featurizer = get_spec_featurizer(params['spectra_view'], params)
    # mol_featurizer = get_mol_featurizer(params['molecule_view'], params)
    # #Assign precompute dataset
    # dataset = get_test_ms_dataset(mol_view= params['molecule_view'], mol_featurizer=mol_featurizer,params= params)

    # # Init data module for precomputation and testing
    # collate_fn = partial(ContrastiveDataset.collate_fn, spec_enc=params['spec_enc'], spectra_view=params['spectra_view'], stage= params['stage'])
    
    # # Debug: check dataset length
    # #print(f"DEBUG: Dataset length: {len(dataset)}")
    # #if len(dataset) > 0:
    #    # print(f"DEBUG: First dataset item type: {type(dataset[0])}")
    
    # data_module = PredictDataModule(
    #     dataset=dataset,
    #     collate_fn=collate_fn,
    #     batch_size=1, # already a list of lists so each batch receives a list
    #     num_workers=params['num_workers']
    # )

    # model = get_model(params= params, stage=Stage(params['stage']), model=params['model'])
    # #model.df_test_path = params['df_test_path']
    # model.faiss_index_path = params['faiss_index_dir']
    
    # # Init trainer
    # trainer = Trainer(
    #     accelerator=params['accelerator'],
    #     devices=params['devices'],
    #     default_root_dir=params['experiment_dir']
    # )

    # # Prepare data module to predict
    # data_module.prepare_data()
    # data_module.setup(stage="predict")
    
    # # Precompute Faiss index via predict
    # outputs = trainer.predict(model, datamodule=data_module)
    # print(f"Finished precomputing Faiss index. Saving to disk path: {params['faiss_index_dir']}")
    # print(f"outputs length: {len(outputs)}")

    
    # Compute the FAISS register index and the dir for ID to index mapping
    
    # 1. create a mapping from ID to index file path
    centroids = sorted(f for f in os.listdir(params['faiss_index_dir']) if f.endswith(".faiss") and "_" in f)
    # 2. get centroid masses
    centroid_masses = [float(name.split("_")[1].replace(".faiss", "")) for name in centroids]
    # 3. create a faiss index from centoid mass to ID
    centroid_index = faiss.IndexFlatL2(1)
    centroid_index.add(np.array(centroid_masses, dtype="float32").reshape(-1, 1))
    # 4. save the centroid index and the mapping to disk
    if not os.path.exists(f"{params['faiss_index_dir']}/Registry"):
        os.makedirs(f"{params['faiss_index_dir']}/Registry")
    faiss.write_index(centroid_index, f"{params['faiss_index_dir']}/Registry/centroid_index.faiss")
    with open(f"{params['faiss_index_dir']}/Registry/id_to_index_mapping.json", "w") as f:
        json.dump(centroids, f)

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
    elif params['checkpoint_pth_mol_enc'] != '':
        pass
    else:
        print("No model chekpoint and mol encoder checkpoint provided. Using the checkpoint in the experiment directory")
        for f in os.listdir(exp_dir):
            if f.endswith("ckpt") and f.startswith("epoch") and args.checkpoint_choice in f:
                checkpoint_path = os.path.join(exp_dir, f)
                params['checkpoint_pth'] = checkpoint_path
                break
        if not params['checkpoint_pth']:
            raise ValueError("No checkpoint provided. Please provide a checkpoint path.")
    
    if args.candidates_pth:
        params['candidates_pth'] = args.candidates_pth
    if args.faiss_index_dir:
        params['faiss_index_dir'] = args.faiss_index_dir
    if not params['faiss_index_dir']:
        if params['candidates_pth'] == None:
            TypeError("Please provide a path to the candidate molecules file and a path to save the faiss index.")
        else:
            params['faiss_index_dir'] = os.path.join(os.getcwd(), "indexes")
    print("FAISS index file path: ", params['faiss_index_dir'])
    start_time = time.time()
    main(params)
    end_time = time.time()
    print(f"Total precomputation time: {end_time - start_time:.2f} seconds)")