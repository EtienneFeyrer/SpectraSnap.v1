import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd

def pad_graph_nodes(mol_enc, g_n_nodes):
    """
    Args:
        mol_enc: 2D tensor of shape (sum_nodes, D)
                 Node embeddings for each molecule.
        g_n_nodes: list[int]  Number of nodes per graph (len = B)

    Returns:
        padded: (B, max_nodes, D) tensor
        mask:   (B, max_nodes) bool tensor, True for valid nodes
    """

    # Already concatenated: shape (sum_nodes, D)
    B = len(g_n_nodes)
    D = mol_enc.shape[1]
    max_nodes = max(g_n_nodes)
    padded = mol_enc.new_zeros((B, max_nodes, D))
    mask = torch.zeros((B, max_nodes), dtype=torch.bool, device=mol_enc.device)

    idx = 0
    for i, n in enumerate(g_n_nodes):
        padded[i, :n] = mol_enc[idx:idx+n]
        mask[i, :n] = True
        idx += n
    return padded, mask

def faiss_to_unified_format(faiss_results, unlabeled_spectrum, list_of_lists_of_candidates):
    """Convert faiss results (list of lists of candidate indices) 
    to unified format (list of lists of candidate dicts)
    Input: faiss_results: Column1 :list of list of candidate indices Column2: scores of candidates,
      unlabeled_spectrum: Spectra dataset(to get identifiers),
    list_of_lists_of_candidates: list of list of candidate dicts (to get candidate smiles)
    Order has to be preserved to match spectra with their candidates
    Output: unified_results: Column1[identifier] : spectrum identifier, Column2[candidates]: list of candidates(smiles),
      Column3[scores]: list of scores, Column4[labels]: list of istarget (1st smile in cand), """

    unified_results = []
    for spec_idx, cand_scores in enumerate(faiss_results):
        spectrum = getattr(unlabeled_spectrum, "spectra", [None] * len(faiss_results))[spec_idx]
        spectrum_id = spectrum.metadata.get("identifier", spec_idx) if spectrum is not None else spec_idx

        candidate_group = list_of_lists_of_candidates[spec_idx]
        if isinstance(candidate_group, dict):
            candidate_group = candidate_group.get("cand_smiles", candidate_group.get("candidates", []))
        target_smile = candidate_group[0] if candidate_group else None

        candidates = []
        scores = []
        labels = []

        # Iterate over FAISS results in the order they are provided (do NOT sort)
        if isinstance(cand_scores, dict):
            for idx in cand_scores:
                score = cand_scores[idx]
                cand_idx = int(idx)
                candidates.append(candidate_group[cand_idx])
                scores.append(score)
                labels.append(candidate_group[cand_idx] == target_smile)
        else:
            for idx in cand_scores:
                cand_idx = int(idx)
                candidates.append(candidate_group[cand_idx])
                scores.append(None)
                labels.append(candidate_group[cand_idx] == target_smile)

        unified_results.append({
            "identifier": spectrum_id,
            "candidates": candidates,
            "scores": scores,
            "labels": labels,
        })

    return pd.DataFrame(unified_results, columns=["identifier", "candidates", "scores", "labels"])
