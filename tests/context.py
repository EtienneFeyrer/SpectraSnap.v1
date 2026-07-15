# TODO: add the context for testing if neccessary
import sys
from pathlib import Path
# all test execute from the project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jestr.data.datasets as datasets
import jestr.utils.eval as eval_utils


# FAISS test

import argparse
import datetime
import sys
import os 
import time
import json
import torch

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rdkit import RDLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from massspecgym.models.base import Stage

from jestr.data.data_module import TestDataModule, PredictDataModule
from jestr.data.datasets import ContrastiveDataset
from jestr.utils.data import get_spec_featurizer, get_mol_featurizer, get_test_ms_dataset
from jestr.utils.models import get_model
from jestr.utils.general import faiss_to_unified_format

from jestr.definitions import TEST_RESULTS_DIR
import yaml
from functools import partial
import faiss
from jestr.data.datasets import SpecDataset
from jestr.data.datasets import PrecomputeCandDataset