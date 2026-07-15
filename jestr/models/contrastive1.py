import typing as T
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
import numpy as np
import os
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.models.base import Stage
from massspecgym import utils
import faiss
from jestr.utils.loss import contrastive_loss, cand_spec_sim_loss, fp_loss, cons_spec_loss
import jestr.utils.models as model_utils

import torch.nn.functional as F

class ContrastiveModel(RetrievalMassSpecGymModel):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.spec_enc_model = model_utils.get_spec_encoder(self.hparams.spec_enc, self.hparams)
        self.mol_enc_model = model_utils.get_mol_encoder(self.hparams.mol_enc, self.hparams)
            
        self.spec_view = self.hparams.spectra_view
        
        # result storage for testing results
        self.result_dct = defaultdict(lambda: defaultdict(list))
                    
    def forward(self, batch, stage):
        g = batch['cand'] if stage == Stage.TEST else batch['mol']
        
        spec = batch[self.spec_view]
        n_peaks = batch['n_peaks'] if 'n_peaks' in batch else None
        spec_enc = self.spec_enc_model(spec, n_peaks)

        mol_enc = self.mol_enc_model(g)

        return spec_enc, mol_enc

    def compute_loss(self, batch: dict, spec_enc, mol_enc):
        loss = 0
        losses = {}
        contr_loss, _, _ = contrastive_loss(spec_enc, mol_enc, self.hparams.contr_temp)
        
        loss+=contr_loss
        losses['loss'] = loss 

        return losses
    
    def step(
        self, batch: dict, stage= Stage.NONE):
        
        # Compute spectra and mol encoding
        spec_enc, mol_enc = self.forward(batch, stage)

        if stage == Stage.TEST:
            return dict(spec_enc=spec_enc, mol_enc=mol_enc)

        # Calculate loss
        losses = self.compute_loss(batch, spec_enc, mol_enc)

        return losses

    def on_batch_end(self, outputs, batch: dict, batch_idx: int, stage: Stage) -> None:
        # total loss
        self.log(
            f'{stage.to_pref()}loss',
            outputs['loss'],
            batch_size=len(batch['identifier']),
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            # on_step=True
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if batch['stage'] == Stage.PRECOMPUTE:
            os.makedirs("indexes", exist_ok=True)
            index = self.step(batch, stage=Stage.PRECOMPUTE)
            faiss.write_index(index, f"indexes/index_{batch_idx}.faiss")
            return None

    def test_step(self, batch, batch_idx):
        # Unpack inputs
        identifiers = batch['identifier']
        cand_smiles = batch['cand_smiles']
        
        #print(f"DEBUG test_step: batch_idx={batch_idx}, identifiers={len(identifiers)}, cand_smiles={len(cand_smiles)}")
        
        id_to_ct = defaultdict(int)
        for i in identifiers: id_to_ct[i]+=1
        batch_ptr = torch.tensor(list(id_to_ct.values()))

        outputs = self.step(batch, stage=Stage.TEST)
        spec_enc = outputs['spec_enc']
        mol_enc = outputs['mol_enc']

        # Calculate scores
        indexes = utils.batch_ptr_to_batch_idx(batch_ptr)
        
        scores = nn.functional.cosine_similarity(spec_enc, mol_enc)
        scores = torch.split(scores, list(id_to_ct.values()))

        cand_smiles = utils.unbatch_list(batch['cand_smiles'], indexes)
        labels = utils.unbatch_list(batch['label'], indexes)
        
        #print(f"DEBUG test_step result: {len(list(id_to_ct.keys()))} identifiers, scores split into {len(scores)} groups")
        
        return dict(identifiers=list(id_to_ct.keys()), scores=scores, cand_smiles=cand_smiles, labels=labels)
    
    def on_test_batch_end(self, outputs, batch: dict, batch_idx: int, stage: Stage = Stage.TEST) -> None:
        
        # Debug: check if outputs are populated
       # print(f"DEBUG on_test_batch_end: batch_idx={batch_idx}, outputs keys={outputs.keys() if outputs else 'None'}")
       # if outputs and 'identifiers' in outputs:
            #print(f"DEBUG: {len(outputs['identifiers'])} identifiers in batch")
        
        # save scores
        for i, cands, scores, l in zip(outputs['identifiers'], outputs['cand_smiles'], outputs['scores'], outputs['labels']):
            self.result_dct[i]['candidates'].extend(cands)
            self.result_dct[i]['scores'].extend(scores.cpu().tolist())
            self.result_dct[i]['labels'].extend([x.cpu().item() for x in l])
            
    def _compute_rank(self, scores, labels):
        if not any(labels):
            return -1
        scores = np.array(scores)
        target_score = scores[labels][0]
        rank = np.count_nonzero(scores >=target_score)
        return rank
    
    def _sort_candidates(self, scores, candidates):
        scores = np.array(scores)
        candidates = np.array(candidates)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_candidates = candidates[sorted_indices]
        sorted_scores = scores[sorted_indices]
        return sorted_candidates.tolist(), sorted_scores.tolist()

    def _sort_candidates_with_labels(self, scores, candidates, labels):
        scores = np.array(scores)
        candidates = np.array(candidates)
        labels = np.array(labels, dtype=bool)

        sorted_indices = np.argsort(scores)[::-1]
        sorted_candidates = candidates[sorted_indices].tolist()
        sorted_scores = scores[sorted_indices].tolist()
        sorted_labels = labels[sorted_indices].tolist()

        target_candidate = candidates[labels][0] if labels.any() else None
        rank = int(np.argmax(sorted_labels)) + 1 if any(sorted_labels) else -1

        return sorted_candidates, sorted_scores, sorted_labels, target_candidate, rank
    
    def on_test_epoch_end(self) -> None:
        #print(f"DEBUG on_test_epoch_end: result_dct has {len(self.result_dct)} identifiers")
       # if len(self.result_dct) == 0:
        #    print("WARNING: result_dct is empty! No batches were processed.")
        
        self.df_test = pd.DataFrame.from_dict(self.result_dct, orient='index').reset_index().rename(columns={'index': 'identifier'})
       # print(f"DEBUG: Created DataFrame with shape {self.df_test.shape}")
        #print(f"DEBUG: df_test_path = {self.df_test_path}")
        # print(f"DEBUG: df_test columns = {list(self.df_test.columns)}")
        # print(f"DEBUG: df_test first 3 rows:\n{self.df_test.head(3)}")
        # print(f"DEBUG: df_test dtypes:\n{self.df_test.dtypes}")

        # Compute rank and sort candidates/scores for all paths
        # This works for both 'smiles' and 'identifier' candidate_file_key since candidates are stored in both cases
        (
            self.df_test['sorted_candidates'],
            self.df_test['sorted_scores'],
            self.df_test['sorted_labels'],
            self.df_test['target_candidate'],
            self.df_test['rank'],
        ) = zip(*self.df_test.apply(
            lambda row: self._sort_candidates_with_labels(
                row['scores'], row['candidates'], row['labels']
            ),
            axis=1,
        ))
        self.df_test['hit_rate@1'] = (self.df_test['rank'] <= 1).astype(int)
        self.df_test['hit_rate@5'] = (self.df_test['rank'] <= 5).astype(int)
        self.df_test['hit_rate@20'] = (self.df_test['rank'] <= 20).astype(int)
        self.df_test = self.df_test[
            [
                'identifier',
                'sorted_candidates',
                'sorted_scores',
                'sorted_labels',
                'target_candidate',
                'rank',
                'hit_rate@1',
                'hit_rate@5',
                'hit_rate@20',
            ]
        ]
        
        # print(f"DEBUG: Before save, df_test shape = {self.df_test.shape}")
        # print(f"DEBUG: Saving to {self.df_test_path}")
        self.df_test.to_pickle(self.df_test_path)
        
        # Verify file was written
        import os
        if os.path.exists(self.df_test_path):
            file_size = os.path.getsize(self.df_test_path)
            # print(f"DEBUG: File saved successfully. File size: {file_size} bytes")
        else:
            print(f"ERROR: File {self.df_test_path} was not created!")

    def get_checkpoint_monitors(self) -> T.List[dict]:
        monitors = [
            {"monitor": f"{Stage.TRAIN.to_pref()}loss", "mode": "min", "early_stopping": False}, # monitor train loss
        ]
        return monitors
    
    def _update_loss_weights(self)-> None:
        if self.hparams.loss_strategy == 'linear':
            for loss in self.loss_wts:
                self.loss_wts[loss] += self.loss_updates[loss]
        elif self.hparams.loss_strategy == 'manual':
            for loss in self.loss_wts:
                if self.current_epoch in self.loss_updates[loss]:
                    self.loss_wts[loss] = self.loss_updates[loss][self.current_epoch]

    def on_train_epoch_end(self) -> None:
        self._update_loss_weights()      