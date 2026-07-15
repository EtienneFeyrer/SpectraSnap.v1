from torch.utils.data.dataloader import DataLoader
from massspecgym.data.data_module import MassSpecDataModule
from jestr.data.datasets import ContrastiveDataset
from functools import partial
from massspecgym.models.base import Stage

#TODO: extend this TestDataModule to support unlabeled data for precomputing
# and online testing
# think about the stages and weather they are required or not
class TestDataModule(MassSpecDataModule):
    def __init__(
            self,
            collate_fn,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.collate_fn = collate_fn

    def prepare_data(self):
        pass
    #TODO: remove redundant variables
    def setup(self, stage=None):
        if stage == "test":
            self.test_dataset = self.dataset
        elif stage == "predict":
            self.test_dataset = self.dataset
        elif stage == "online_test":
            self.test_dataset = self.dataset
        else:
            raise Exception("Data module supports test set only")
        
    def predict_dataloader(self):
        #TODO this has to be changed to support the precompute
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return None
    
    def val_dataset(self):
        return None

class ContrastiveDataModule(MassSpecDataModule):
    def __init__(
            self,
            collate_fn,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.collate_fn = collate_fn
        self.regularization_flag = False
             
    def train_dataloader(self):
        self.train_contrastive_dataset = ContrastiveDataset(self.train_dataset)

        return DataLoader(self.train_contrastive_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          drop_last=False,
                          collate_fn=partial(self.collate_fn, stage=Stage.TRAIN),
                          )

    def val_dataloader(self):
        self.val_contrastive_dataset = ContrastiveDataset(self.val_dataset)

        return DataLoader(self.val_contrastive_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          drop_last=False,
                          collate_fn=partial(self.collate_fn, stage=Stage.VAL))

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.dataset.collate_fn,
        )
    
class PredictDataModule(MassSpecDataModule):
    def __init__(
            self,
            collate_fn,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.collate_fn = collate_fn

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if stage == "predict":
            self.predict_dataset = self.dataset
        else:
            raise Exception("Data module supports predict set only")
        
    def predict_dataloader(self):
        #TODO this has to be changed to support the precompute
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return None
    
    def val_dataset(self):
        return None