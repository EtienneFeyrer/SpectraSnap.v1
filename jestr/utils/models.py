from jestr.models.faiss import FaissModel, PCAModel, RetrievalFaissModel
from jestr.models.spec_encoder import SpecEncMLP_BIN
from jestr.models.mol_encoder import MolEnc
from jestr.models.encoders import MLP
from jestr.models.interaction_model import INTER_MLP2
from jestr.models.contrastive import ContrastiveModel
from massspecgym.models.base import Stage
import torch


def get_spec_encoder(spec_enc: str, args):
    return {"MLP_BIN": SpecEncMLP_BIN}[spec_enc](args)


def get_mol_encoder(mol_enc: str, args):
    return {'GNN': MolEnc}[mol_enc](args, in_dim=78)


def get_inter_model(inter_model: str, args):
    return {'INTER_MLP2': INTER_MLP2}[inter_model](args)


def load_weights(model, pretrained_model):
    """Load checkpoint safely on CPU and strip 'module.' prefixes."""
    loaded = torch.load(pretrained_model, map_location="cpu")

    if isinstance(loaded, dict) and loaded.get('model_state_dict', None) is not None:
        loaded = loaded['model_state_dict']

    state_dict = {}
    for key, value in loaded.items():
        if key.startswith("module."):
            state_dict[key[7:]] = value
        else:
            state_dict[key] = value
    return state_dict


def get_model(stage: Stage, model: str, params):
    stage = Stage(stage)

    # -------------------------
    # PRECOMPUTE STAGE
    # -------------------------
    if stage == Stage.PRECOMPUTE:
        if params['faiss_model'] == 'Flat':
            model = FaissModel(**params)
        elif params['faiss_model'] == 'PCA Flat':
            model = PCAModel(**params)

        # Load molecular encoder safely
        if params.get('checkpoint_pth_mol_enc'):
            state = torch.load(params['checkpoint_pth_mol_enc'], map_location="cpu")
            model.mol_enc_model.load_state_dict(state, strict=False)
            print("Loaded molecular encoder from checkpoint (CPU-safe)")

    # -------------------------
    # ONLINE TEST STAGE
    # -------------------------
    elif stage == Stage.ONLINE_TEST:
        model = RetrievalFaissModel(**params)

        if params.get('checkpoint_pth_spec_enc'):
            state = torch.load(params['checkpoint_pth_spec_enc'], map_location="cpu")
            model.spec_enc_model.load_state_dict(state)
            print("Loaded spectral encoder from checkpoint (CPU-safe)")

    # -------------------------
    # TRAIN / TEST / VAL STAGES
    # -------------------------
    elif stage in (Stage.TEST, Stage.TRAIN, Stage.VAL):
        model = ContrastiveModel(**params)

        # Spectral encoder
        if params.get('checkpoint_pth_spec_enc'):
            state = torch.load(params['checkpoint_pth_spec_enc'], map_location="cpu")
            model.spec_enc_model.load_state_dict(state)
            print("Loaded spectral encoder from checkpoint (CPU-safe)")

        # Molecular encoder
        if params.get('checkpoint_pth_mol_enc'):
            state = torch.load(params['checkpoint_pth_mol_enc'], map_location="cpu")
            model.mol_enc_model.load_state_dict(state, strict=False)
            print("Loaded molecular encoder from checkpoint (CPU-safe)")

        # Interaction model
        if params.get('checkpoint_pth_inter_model'):
            state = load_weights(model.inter_model, params['checkpoint_pth_inter_model'])
            model.inter_model.load_state_dict(state)
            print("Loaded interaction model from checkpoint (CPU-safe)")

    else:
        raise Exception(f"Stage {stage} with model {model} not implemented.")

    # -------------------------
    # FULL MODEL CHECKPOINT (Lightning)
    # -------------------------
    if params.get('checkpoint_pth'):
        model = type(model).load_from_checkpoint(
            params['checkpoint_pth'],
            log_only_loss_at_stages=params['log_only_loss_at_stages'],
            df_test_path=params['df_test_path'],
            map_location="cpu"   # ensure CPU-safe loading
        )
        print("Loaded full Lightning model checkpoint (CPU-safe)")

    return model

