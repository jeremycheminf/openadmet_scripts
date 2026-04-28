import os
import itertools
import warnings
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, nn, models
from chemprop.models.model import MPNN
from chemprop.models.utils import save_model, load_model
from sklearn.model_selection import train_test_split
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping


class ChemPropChemeleonWrapper:
    def __init__(self, y_name: str):
        self.y_name = y_name
        self.model = None
        self.scaler = None

    def fit(self, train: pd.DataFrame, num_epochs: int = 20, accelerator: str = "gpu"):
        if not os.path.exists(os.environ["CHEMPROP_CACHE_DIR"] + "chemeleon_mp.pt"):
            chemeleon_path = urlretrieve(
                r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                "chemeleon_mp.pt",
            )[0]
        else:
            chemeleon_path = os.environ["CHEMPROP_CACHE_DIR"] + "chemeleon_mp.pt"

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        chemeleon_mp = torch.load(chemeleon_path, weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        mp.load_state_dict(chemeleon_mp["state_dict"])

        train, val = train_test_split(train, test_size=0.2)
        train_pt = [
            data.MoleculeDatapoint.from_smi(smi, [y])
            for smi, y in train[["SMILES", self.y_name]].values
        ]
        val_pt = [
            data.MoleculeDatapoint.from_smi(smi, [y])
            for smi, y in val[["SMILES", self.y_name]].values
        ]

        train_dset = data.MoleculeDataset(train_pt, featurizer)
        self.scaler = train_dset.normalize_targets()
        val_dset = data.MoleculeDataset(val_pt, featurizer)
        val_dset.normalize_targets(self.scaler)

        train_loader = data.build_dataloader(train_dset, num_workers=0)
        val_loader = data.build_dataloader(val_dset, num_workers=0, shuffle=False)

        ffn = nn.RegressionFFN(
            input_dim=mp.output_dim,
            output_transform=nn.UnscaleTransform.from_standard_scaler(self.scaler),
        )

        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]
        self.model = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=metric_list)

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator=accelerator,
            devices=1,
            max_epochs=num_epochs,
        )
        trainer.fit(self.model, train_loader, val_loader)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        test_pts = [data.MoleculeDatapoint.from_smi(smi, [0]) for smi in test["SMILES"]]

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_dset = data.MoleculeDataset(test_pts, featurizer)
        test_loader = data.build_dataloader(test_dset, num_workers=0, shuffle=False)

        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                accelerator="gpu",
                devices=1,
            )
            predictions = trainer.predict(self.model, test_loader)

        pred = np.array(list(itertools.chain(*predictions))).flatten()
        return pred

    def validate(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        num_epochs: int = 20,
        accelerator: str = "gpu",
    ):
        self.fit(train, num_epochs=num_epochs, accelerator=accelerator)
        return self.predict(test)

    def save_model(self, path: str):
        if self.model is not None:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model = MPNN.load_from_file(path)

df_train = pd.read_csv(
    "hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train.csv"
)
df_test = pd.read_csv(
    "hf://datasets/openadmet/openadmet-expansionrx-challenge-test-data-blinded/expansion_data_test_blinded.csv"
)

prop_ok = ["LogD"]
props_log = [
    "HLM CLint",
    "MLM CLint",
    "KSOL",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
]
props_pk = ["MGMB", "MPPB", "MBPB"]

# log properties and create fake values for 0
def log_prop(df, prop):
    df[prop] = np.where(df[prop] == 0, 0.1, df[prop])
    df[prop] = df[prop].apply(lambda x: np.log10(x))


def pk_prop(df, prop):
    df[prop] = 100 - df[prop]
    df[prop] = np.where(df[prop] == 100, 99.9, df[prop])
    df[prop] = df[prop].apply(lambda x: np.log10(x / (100 - x)))


for prop in props_log:
    log_prop(df_train, prop)

for prop in props_pk:
    pk_prop(df_train, prop)
	
# Run
for prop in props_log + props_pk:
    Y_COL = prop
    df_train_chemprop = df_train.dropna(subset=[Y_COL])
    model_split = ChemPropChemeleonWrapper(Y_COL)
    pred_split = model_split.validate(df_train_chemprop, df_test)
    df_test[Y_COL] = pred_split
    df_test.to_csv("predictions.csv", index=False)

# prepare for submission
def unlog_prop(df, prop):
    df[prop] = df[prop].apply(lambda x: 10**x)


def unpk_prop(df, prop):
    df[prop] = df[prop].apply(lambda x: 10**x)
    df[prop] = df[prop] * 100 / (1 + df[prop])
    df[prop] = 100 - df[prop]

df_test_sub = df_test.copy()
for prop in props_log:
    unlog_prop(df_test_sub, prop)
for prop in props_pk:
    unpk_prop(df_test_sub, prop)
	

# save files
df_test_sub.to_csv("submissions_chemeleon_stl.csv", index=False)
