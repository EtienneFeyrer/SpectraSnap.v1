""" This file contains two models for the FAISS workflow:
1. FaissModel 
2. RetrievalFaissModel """
from abc import ABC, abstractmethod
import pytorch_lightning as pl
from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
import faiss
import torch
import os
import jestr.utils.models as model_utils
from collections import defaultdict
import numpy as np

class FaissModel(pl.LightningModule, ABC):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mol_enc_model = model_utils.get_mol_encoder(self.hparams.mol_enc, self.hparams)

    def predict_step(self, batch: dict):
        '''Input: batch of candidate molecules.
        Output: Faiss Index on cpu build on encodings
        of the candidate molecules.'''
        group_id = batch['group_id'][0]
        #use the forward method to get molecule encodings
        encodings = self.forward(batch)
        # normalize encodings for cosine similarity L2 normalization
        enc = torch.nn.functional.normalize(encodings, p=2, dim=1)
        # convert to numpy array for faiss 
        enc = enc.detach().cpu().numpy().astype('float32')
        # build faiss index on gpu and move to cpu
        d = enc.shape[1]
        cpu_index = faiss.IndexFlatIP(d)
        cpu_index.add(enc)
        #print(f"Built FAISS index for batch {group_id} saving to {self.hparams['faiss_index_dir']}/index_{group_id}.faiss")
        os.makedirs(self.hparams['faiss_index_dir'], exist_ok=True)
        faiss.write_index(cpu_index, f"{self.hparams['faiss_index_dir']}/index_{group_id}.faiss")
        #return cpu_index

    def forward(self, batch: dict):
        '''Returns molecule embeddings given a batch of 
        candidate molecules.'''
        g = batch['cand']
        return self.mol_enc_model(g)
    
    # def on_predict_epoch_end(self):
    #     '''Input: Outputs from each batch in this case a list of faiss indexes.
    #     Output: None, but saves the indexes to disk.'''
    #     outputs = self._output
    #     #check if directory exists and create if not
    #     os.makedirs(self.hparams['faiss_index_dir'], exist_ok=True)
    #     #save each index file to disk
    #     for i, index in enumerate(outputs): #outputs are ordered according to batch index
    #         faiss.write_index(index, f"{self.hparams['faiss_index_dir']}/index_{i}.faiss")
    #     return None
    

class RetrievalFaissModel(pl.LightningModule, ABC):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.spec_enc_model = model_utils.get_spec_encoder(self.hparams.spec_enc, self.hparams)
        self.spec_view = self.hparams.spectra_view
        self.faiss_index_path = getattr(self.hparams, "faiss_index_dir", None)
        
        # result storage for testing results
        #TODO: not needed
        self.result_dct = defaultdict(lambda: defaultdict(list))
        
        # cache for FAISS indexes to avoid repeated disk reads
        self._index_cache = {}
        # Keep retrieval on CPU; FAISS uses OpenMP threads for the search.
        faiss.omp_set_num_threads(max(1, os.cpu_count() or 1))

    def predict_step(self, batch: dict, stage: Stage = Stage.NONE):
        '''Input: batch with spectrum view.
        Output: Dictionary containing k entries where key is the k 
        and value is the distance to the k'st neighbor.'''
        # use the forward method to get molecule encodings
        encoding = self.forward(batch, stage)
        # normalize encodings for cosine similarity L2 normalization
        enc = torch.nn.functional.normalize(encoding, p=2, dim=1)
        # convert to numpy array for faiss 
        enc = enc.detach().cpu().numpy().astype('float32')
        # search faiss index and PCA if applicable on CPU (cached for performance)
        spectrum_ids = batch['spec_id']
        results = []
        
        for i, spectrum_id in enumerate(spectrum_ids):
            enc_i = enc[i:i+1] # shape (1, d) for faiss search
            spectrum_id = int(spectrum_id)
            
            # load index from cache or disk
            if spectrum_id not in self._index_cache:
                # load PCA model and apply to encoding before lookup
                if self.hparams['faiss_model'] == "PCA Flat":
                    #print (f"Loading PCA Model from: {self.hparams['pca_dir']}/pca_model{str(spectrum_id)}.pca")
                    pca = faiss.read_VectorTransform(f"{self.hparams['pca_dir']}/pca_model{str(spectrum_id)}.pca")
                    enc_i = pca.apply_py(enc_i)
                    #print(f"Applied transform query has dimension {enc_i.shape}")
                #print (f"Loading FAISS index from: {self.faiss_index_path}/index_{spectrum_id}.faiss")
                index_path = f"{self.faiss_index_path}/index_{spectrum_id}.faiss"
                self._index_cache[spectrum_id] = faiss.read_index(index_path)
            
            index = self._index_cache[spectrum_id]
            distances, indices = index.search(enc_i, self.hparams.k)
            results.append({int(idx): float(dist) for idx, dist in zip(indices[0], distances[0])})
        
        # return single dict if batch size is 1, else return list
        return results[0] if len(results) == 1 else results 
    
    def forward(self, batch: dict, stage: Stage = Stage.NONE):
        '''Input: batch of spectra
        Otput: embeddinga of the spectra'''
        spec_key = self.spec_view[0] if isinstance(self.spec_view, (list, tuple)) else self.spec_view
        spec = batch[spec_key]
        n_peaks = batch['n_peaks'] if 'n_peaks' in batch else None
        spec_enc = self.spec_enc_model(spec, n_peaks)
        return spec_enc
    
    
class PCAModel(pl.LightningModule, ABC):
    '''Model computes embeddings of candidate molecules 
    and stores them as embeddings_batch{i}.npy. 
    PCA is fit on epoch end when all embeddings are collected.'''
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mol_enc_model = model_utils.get_mol_encoder(self.hparams.mol_enc, self.hparams)
        self.pca_dir = getattr(self.hparams, "pca_dir", None)
        self.pca_dim = getattr(self.hparams, "pca_dim", None)
        self.index_dir = getattr(self.hparams, "faiss_index_dir", None)
        #self.pca = model_utils.get_pca(self.hparams.pca, self.hparams)

    def predict_step(self, batch: dict):
        '''Input: batch of candidate molecules.
        Output: Faiss Index on cpu build on encodings
        of the candidate molecules.'''
        # use the forward method to get molecule encodings
        encodings = self.forward(batch)
        batch_id = batch['group_id']
        group_id = batch_id[0]
        #1. transfer encodings to cpu
        encodings = encodings.detach().cpu().numpy()
        
        #2. Compute PCA on encodings with whitening if specified
        d_in = int(encodings.shape[1])
        samples = encodings.shape[0]
        d_out = int(samples)  # Convert percentage to actual number of dimensions
        if self.hparams.get('whitening', False):
            pca = faiss.PCAMatrix(d_in, d_out, float(-0.5))
        else:
            pca = faiss.PCAMatrix(d_in, d_out)
        pca.train(encodings)
        pca_encodings = pca.apply_py(encodings)
        #print(f"dimension of encodinsgs after PCA: {encodings.shape} reduced to {pca_encodings.shape}")

        #3. create a FAISS index on the PCA encodings
        res = faiss.StandardGpuResources()
        d = pca_encodings.shape[1]
        gpu_index = faiss.GpuIndexFlatIP(res, d)
        gpu_index.add(pca_encodings.astype('float32'))
        cpu_index = faiss.index_gpu_to_cpu(gpu_index)
        #print(f"Built PCA FAISS index for batch {group_id} saving to {self.index_dir}/index_{group_id}.faiss")
        
        #4. save Faiss index to disk
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(cpu_index, f"{self.index_dir}/index_{group_id}.faiss")
        #5. save PCA model to disk
        os.makedirs(self.pca_dir, exist_ok=True)
        faiss.write_VectorTransform(pca, f"{self.pca_dir}/pca_model{group_id}.pca")

    def forward(self, batch: dict):
        '''Returns molecule embeddings given a batch of 
        candidate molecules.'''
        g = batch['cand']
        return self.mol_enc_model(g)



        
       

# class PCAFaissModel(pl.LightningModule, ABC):
#     '''Model computes PCA embeddings of candidate molecules 
#     and builds a Faiss index on the PCA embeddings.'''
#     def __init__(
#         self,
#         **kwargs
#     ):
#         super().__init__()
#         self.save_hyperparameters()
#         self.mol_enc_model = model_utils.get_mol_encoder(self.hparams.mol_enc, self.hparams)
#         self.pca_dir = getattr(self.hparams, "PCA_dir", None)
#         self.index_dir = getattr(self.hparams, "faiss_index_dir", None)
#         # Load PCA model from disk
#         pca_path = f"{self.pca_dir}/pca_model.pca"
#         if not os.path.exists(pca_path):
#             raise FileNotFoundError(f"PCA model not found at {pca_path}. Please run the PCA model first to fit and save the PCA model.")
#         self.pca = faiss.read_VectorTransform(pca_path)

#     def predict_step(self, batch: dict):
#         '''Input: batch id
#         Output: Faiss Index on cpu build on PCA encodings'''
#         batch_id = batch['group_id']
#         group_id = int(batch_id)
#         encodings_path = f"{self.index_dir}/embeddings_batch{group_id}.npy"
#         if not os.path.exists(encodings_path):
#             raise FileNotFoundError(f"Encodings for batch {group_id} not found at {encodings_path}. Please run the PCA model first to compute and save the batch encodings.")
#         encodings = np.load(encodings_path)
#         # 1. apply PCA to encodings
#         pca_encodings = self.pca.apply_py(encodings)
#         # 2. build faiss index on PCA encodings
#         res = faiss.StandardGpuResources()
#         d = pca_encodings.shape[1]
#         gpu_index = faiss.GpuIndexFlatIP(res, d)
#         gpu_index.add(pca_encodings)
#         cpu_index = faiss.index_gpu_to_cpu(gpu_index)
#         print(f"Built PCA FAISS index for batch {group_id} saving to {self.index_dir}/index_{group_id}.faiss")
#         os.makedirs(self.index_dir, exist_ok=True)
#         faiss.write_index(cpu_index, f"{self.index_dir}/pca_index_{group_id}.faiss")