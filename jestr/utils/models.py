from jestr.models.faiss import FaissModel, PCAModel, RetrievalFaissModel
from jestr.models.spec_encoder import SpecEncMLP_BIN
from jestr.models.mol_encoder import MolEnc
from jestr.models.encoders import MLP
from jestr.models.contrastive import ContrastiveModel
from massspecgym.models.base import Stage
import torch

def get_spec_encoder(spec_enc:str, args):
    return {"MLP_BIN": SpecEncMLP_BIN}[spec_enc](args)

def get_mol_encoder(mol_enc: str, args):
    return {'GNN': MolEnc}[mol_enc](args, in_dim=78)

def get_model(stage: Stage, model:str, params):
    stage = Stage(stage)
    if stage == Stage.PRECOMPUTE:
        if params['faiss_model'] == 'Flat':
            model = FaissModel(**params)
        if params['faiss_model'] == 'PCA Flat':
            model = PCAModel(**params)
        if params['checkpoint_pth_mol_enc'] is not None and params['checkpoint_pth_mol_enc'] !="":
            model.mol_enc_model.load_state_dict(torch.load(params['checkpoint_pth_mol_enc'] , weights_only=True), strict=False)
            print("Loaded molecular encoder from checkpoint")
    
    elif stage == Stage.ONLINE_TEST:
        model = RetrievalFaissModel(**params)
        if params['checkpoint_pth_spec_enc'] is not None and params['checkpoint_pth_spec_enc'] != "":
            model.spec_enc_model.load_state_dict(torch.load(params['checkpoint_pth_spec_enc'], weights_only=True))
            print("Loaded spectral encoder from checkpoint")

    elif stage == Stage.TEST or stage == Stage.TRAIN or stage == Stage.VAL:
        model = ContrastiveModel(**params)
        if params['checkpoint_pth_spec_enc'] is not None and params['checkpoint_pth_spec_enc'] != "":
            model.spec_enc_model.load_state_dict(torch.load(params['checkpoint_pth_spec_enc'], weights_only=True))
            print("Loaded spectral encoder from checkpoint")
        if params['checkpoint_pth_mol_enc'] is not None and params['checkpoint_pth_mol_enc'] !="":
            model.mol_enc_model.load_state_dict(torch.load(params['checkpoint_pth_mol_enc'] , weights_only=True), strict=False)
            print("Loaded molecular encoder from checkpoint")
    else:
        raise Exception(f"Stage {stage} with model {model} not implemented.")
    
    # If checkpoint path is provided, load the model from the checkpoint 
    if params['checkpoint_pth'] is not None and params['checkpoint_pth'] != "":
        model = type(model).load_from_checkpoint(
            params['checkpoint_pth'],
            log_only_loss_at_stages=params['log_only_loss_at_stages'],
            df_test_path=params['df_test_path']
        )
        print("Loaded Model from checkpoint")
    return model