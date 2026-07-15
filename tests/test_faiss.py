"""Integration test for the FAISS scripts.

This test mirrors the shell commands used in the terminal:
1. precompute the FAISS indexes
2. run FAISS inference
3. run regular inference
4. compare the hit rates from the produced result pickles
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy import Null
import yaml

from .context import *


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sample"
PRETRAINED_DIR = ROOT / "pretrained_models" / "MassSpecGym"


def display_hit_rates(dataframe: pd.DataFrame) -> pd.Series:
    working = dataframe.copy()
    working["rank"] = working.apply(
        lambda row: eval_utils.borda_count(
            row["candidates"],
            [row["scores"]],
            eval_utils.get_target(row["candidates"], row["labels"]),
        ),
        axis=1,
    )

    hit_rate_cols = working.apply(
        lambda row: eval_utils.convert_rank_to_hit_rates(row, "rank", top_k=[1, 5, 20]),
        axis=1,
    )
    working = pd.concat([working, hit_rate_cols], axis=1)
    return working[["rank-hit_rate@1", "rank-hit_rate@5", "rank-hit_rate@20"]].mean()

def test_faiss_inference(tmp_path: Path):
    assert torch.cuda.is_available(), "No GPU available"
    # Suppress RDKit warnings and errors
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--param_pth", type=str, default="params_full_inference.yaml")
    # parser.add_argument('--checkpoint_pth', type=str, default='')
    # parser.add_argument('--checkpoint_choice', type=str, default='train', choices=['train', 'val'])
    # parser.add_argument('--df_test_path', type=str, help='result file name')
    # parser.add_argument('--exp_dir', type=str)
    # parser.add_argument('--candidates_pth', type=str)
    # parser.add_argument('--faiss_index_dir', type=str, help='path to precomputed faiss index')

    # args = parser.parse_args([] if "__file__" not in globals() else None)

    # Load parameters file in same directory as this test
    with open("tests/params_test.yml") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    # Experiment directory has to be set before each pipeline
    def set_experiment_start(params, run_name, model):
        "Set experiment directory and make dir on the disk"
        now = datetime.datetime.now().strftime("%Y%m%d")
        exp_dir = str(TEST_RESULTS_DIR / f"{now}_{run_name}")
        os.makedirs(exp_dir, exist_ok=True)
        #print("EXPERIMENT directory: ",exp_dir)
        params['experiment_dir'] = exp_dir
        if not params['df_test_path']:
            params['df_test_path'] = os.path.join(exp_dir, f"result_{params['dataset_pth'].split('/')[-1].split('.')[0]}.pkl")
        else:
            params['df_test_path'] = os.path.join(exp_dir, params['df_test_path'])
        params['model'] = model
        return params    


    # Look for checkpoint
    # if args.candidates_pth:
    #     params['candidates_pth'] = args.candidates_pth
    # if args.df_test_path:
    #     params['df_test_path'] = os.pth.join(exp_dir, args.df_test_path)
    
    # Seed everything
    pl.seed_everything(params['seed'])
        
    # Init paths to data files
    # if params['debug']:
    #     params['dataset_pth'] = "../data/sample/data.tsv"
    #     params['split_pth']=None
    #     params['df_test_path'] = os.path.join(params['experiment_dir'], 'debug_result.pkl')

# -----------------------------------------------------
    # RUN Normal Inference Pipeline
# -----------------------------------------------------

    params = set_experiment_start(params, params['run_name_inference'], params['model_pipeline'][0])
    # If no full checkpoint path provided, try to find a matching experiment checkpoint
    if not params.get('checkpoint_pth'):
        from jestr.definitions import EXPERIMENTS_DIR
        run_name = params.get('run_name_inference') or params.get('run_name')
        found_ckpt = None
        if EXPERIMENTS_DIR.exists():
            for exp in os.listdir(EXPERIMENTS_DIR):
                if exp.endswith("_" + run_name):
                    exp_dir = EXPERIMENTS_DIR / exp
                    for f in os.listdir(exp_dir):
                        if f.endswith("ckpt") and f.startswith("epoch") and 'train' in f:
                            found_ckpt = str(exp_dir / f)
                            break
                if found_ckpt:
                    break
        if found_ckpt:
            params['checkpoint_pth'] = found_ckpt
            print(f"Loaded checkpoint for test from: {found_ckpt}")
        else:
            # No full checkpoint found; fall back to using encoder checkpoints if available
            print("No full model checkpoint found. Will attempt to load encoder weights from params if provided.")
    # Load dataset
    spec_featurizer = get_spec_featurizer(params['spectra_view'], params)
    mol_featurizer = get_mol_featurizer(params['molecule_view'], params)
    # parameters without candidates path
    kwargs = {k:v for k,v in params.items() if k not in ['candidates_pth']}
    kwargs['candidates_pth'] = None
    kwargs['stage'] = Stage.TEST
    dataset = get_test_ms_dataset(
params=kwargs,
spectra_view=params['spectra_view'],
mol_view=params['molecule_view'],
spectra_featurizer=spec_featurizer,
mol_featurizer=mol_featurizer,
)

    # Init data module
    collate_fn = partial(ContrastiveDataset.collate_fn, spec_enc=params['spec_enc'], spectra_view=params['spectra_view'], stage=Stage.TEST)
    data_module = TestDataModule(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=params['batch_size'],
        num_workers=params['num_workers']
    )
    # select specific model for inference
    model = get_model(kwargs['stage'], params['model'], kwargs)
    model.df_test_path = params['df_test_path']
    # If no full checkpoint was set, try loading separate encoder weights if provided
    if not params.get('checkpoint_pth'):
        # spectral encoder
        spec_ck = params.get('checkpoint_pth_spec_enc')
        if spec_ck:
            if hasattr(model, 'spec_enc_model'):
                try:
                    model.spec_enc_model.load_state_dict(torch.load(spec_ck, weights_only=True))
                except TypeError:
                    # older torch versions may not support weights_only
                    model.spec_enc_model.load_state_dict(torch.load(spec_ck))
                print(f"Loaded spectral encoder weights from: {spec_ck}")
            else:
                print("Model has no attribute 'spec_enc_model' to load spectral encoder into.")

        # molecular encoder
        mol_ck = params.get('checkpoint_pth_mol_enc')
        if mol_ck:
            if hasattr(model, 'mol_enc_model'):
                try:
                    model.mol_enc_model.load_state_dict(torch.load(mol_ck, weights_only=True), strict=False)
                except TypeError:
                    model.mol_enc_model.load_state_dict(torch.load(mol_ck), strict=False)
                print(f"Loaded molecular encoder weights from: {mol_ck}")
            else:
                print("Model has no attribute 'mol_enc_model' to load molecular encoder into.")

    # Init trainer
    trainer = Trainer(
        accelerator=params['accelerator'],
        devices=params['devices'],
        default_root_dir=params['experiment_dir']
    )
    # Prepare data module to test
    data_module.prepare_data()
    data_module.setup(stage="test")

    # Test
    print(f"DEBUG: with df test pathh: {params['df_test_path']}")
    trainer.test(model, datamodule=data_module)
    print(f"Finished normal inference. Saving results to disk path: {params['df_test_path']}")
    normal_inference_results = pd.read_pickle(params['df_test_path'])

# ---------------------------------------------------------------
    # RUN Faiss Precompute Pipeline
# ---------------------------------------------------------------
    # Update experiment directory for FAISS precompute
    params = set_experiment_start(params, params['run_name_faiss_precompute'], params['model_pipeline'][1])
    params['stage'] = Stage.PRECOMPUTE
    # Seed everything
    pl.seed_everything(params['seed'])

    # Load dataset
    #spec_featurizer = get_spec_featurizer(params['spectra_view'], params)
    mol_featurizer = get_mol_featurizer(params['molecule_view'], params)
    #Assign precompute dataset
    #kwargs = {k:v for k,v in params.items() if k not in ['spec_featurizer']}
    dataset = get_test_ms_dataset(mol_view= params['molecule_view'], mol_featurizer=mol_featurizer,params= params)

    # Init data module for precomputation and testing
    collate_fn = partial(ContrastiveDataset.collate_fn, spec_enc=params['spec_enc'], spectra_view=params['spectra_view'], stage=params['stage'])
    
    # Debug: check dataset length
    #print(f"DEBUG: Dataset length: {len(dataset)}")
    #if len(dataset) > 0:
       # print(f"DEBUG: First dataset item type: {type(dataset[0])}")
    
    data_module = PredictDataModule(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=1, # already a list of lists so each batch receives a list
        num_workers=params['num_workers']
    )
    # select specific model for FAISS precompute
    #print(f"DEBUG: Faiss model type: {params['faiss_model']}")
    model = get_model(params['stage'], params['model'], params)
    #model.df_test_path = params['df_test_path']
    model.faiss_index_path = params['faiss_index_dir']
    
    # Init trainer
    trainer = Trainer(
        accelerator=params['accelerator'],
        devices=params['devices'],
        default_root_dir=params['experiment_dir']
    )

    # Prepare data module to predict
    data_module.prepare_data()
    data_module.setup(stage="predict")
    
    # Precompute Faiss index via predict
    outputs = trainer.predict(model, datamodule=data_module)
    print(f"Finished precomputing Faiss index. Saving to disk path: {params['faiss_index_dir']}")
    print(f"outputs length: {len(outputs)}")

    #check if directory exists and create if not
    #os.makedirs(params['faiss_index_dir'], exist_ok=True)
    #save each index file to disk
    #for i, index in enumerate(outputs): #outputs are ordered according to batch index
    #    faiss.write_index(index, f"{params['faiss_index_dir']}/index_{i}.faiss")

# ---------------------------------------------------------------
    # RUN Faiss Inference Pipeline
# ---------------------------------------------------------------

    # set Exp dir
    params = set_experiment_start(params, params['run_name_faiss_inference'], params['model_pipeline'][2])
    params['stage'] = Stage.ONLINE_TEST
    print(f"df test path has to contain runname faiss inference: {params['df_test_path']}")
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
    # Set specific model
    model = get_model(params['stage'], params['model'], params)
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
    print(f"Finished precomputing Faiss index. Saving to disk path: {params['faiss_index_dir']}")
    print(f"outputs length: {len(outputs)}")
    #save test results to disk
    output_path = os.path.join(params['experiment_dir'], 'outputs.json')
    with open(output_path, 'w') as f:
        json.dump(outputs, f)
    print(f"Finished testing. Saving results to disk path: {output_path}")
    #create ublabeled datasets for mapping back faiss results to identifiers and candidates
    unlabeled_spectrum = SpecDataset(spectra_pth=params['spectra_pth'], spec_transform=spec_featurizer)
    list_of_lists_of_candidates = PrecomputeCandDataset(raw_pth=params.get('candidates_pth'))

    unified_format = faiss_to_unified_format(outputs, unlabeled_spectrum, list_of_lists_of_candidates)
    print(f"save unified format to disk path: {params['df_test_path']}")
    unified_format.to_pickle(params['df_test_path'])

# ---------------------------------------------------------------
    # CHECK Hit Rates 
# ---------------------------------------------------------------
    # Load the produced test result pickle and display hit rates
    faiss_inference_results = pd.read_pickle(params['df_test_path'])
    normal_hit_rates = display_hit_rates(normal_inference_results)
    faiss_hit_rates = display_hit_rates(faiss_inference_results)
    assert len(normal_hit_rates) == len(faiss_hit_rates), "Hit rate outputs should have the same shape"
    for col in normal_hit_rates.index:
        normal_rate = normal_hit_rates[col]
        faiss_rate = faiss_hit_rates[col]
        print(f"{col}: Normal Inference Hit Rate = {normal_rate:.4f}, FAISS Inference Hit Rate = {faiss_rate:.4f}")
        assert abs(normal_rate - faiss_rate) < 0.05, f"Hit rates for {col} differ by more than 5% between normal and FAISS inference"
