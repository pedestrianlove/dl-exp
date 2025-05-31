import torch
import torch.nn as nn
import polars as pl

from torch.utils.data import Dataset

class CaloriesDataset(Dataset):
    # Class members
    data: pl.DataFrame
    feature_cols: list[str]
    label: str

    # Constructor
    def __init__(self, data_path: str):
        self.feature_cols = [
            'Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
        ]
        self.label = 'Calories'
        self.data = pl.read_csv(data_path).with_columns(
            pl.when(pl.col("Sex") == "male").then(1).otherwise(0).alias("Sex")
        ).select(pl.exclude('id'))

    # Length of the dataset
    def __len__(self):
        return self.data.height

    # Get item by index
    def __getitem__(self, idx: int):
        # 2) grab the entire row as a tuple
        row = self.data.row(idx)  
        #    row will be (feat1, feat2, ..., feat7, label)

        # 3) split out features vs. label by column index
        #    assume feature_cols is a list like ["f1","f2",…,"f7"]
        #    and label_col is the last column name
        feat_vals = row[:len(self.feature_cols)]
        label_val = row[len(self.feature_cols)]

        # 4) to Tensor
        x = torch.tensor(feat_vals, dtype=torch.float32)
        # for regression:
        y = torch.tensor(label_val, dtype=torch.float32)

        return x, y

class CaloriesPrediction(nn.Module):
    # Class members
    input_layer: nn.Linear
    activation: nn.ReLU
    output_layer: nn.Linear

    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.input_layer = nn.Linear(7, 64)
        self.medium_layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.medium_layers(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

class CaloriesTest(Dataset):
    # Class members
    data: pl.DataFrame
    rdata: pl.DataFrame
    feature_cols: list[str]

    # Constructor
    def __init__(self, data_path: str):
        self.feature_cols = [
            'Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
        ]
        self.rdata = pl.read_csv(data_path).with_columns(
            pl.when(pl.col("Sex") == "male").then(1).otherwise(0).alias("Sex")
        )
        self.data = self.rdata.select(pl.exclude('id'))

    # Length of the dataset
    def __len__(self):
        return self.data.height

    # Get item by index
    def __getitem__(self, idx: int):
        # 2) grab the entire row as a tuple
        row = self.data.row(idx)  
        #    row will be (feat1, feat2, ..., feat7, label)

        # 3) split out features vs. label by column index
        #    assume feature_cols is a list like ["f1","f2",…,"f7"]
        #    and label_col is the last column name
        feat_vals = row[:len(self.feature_cols)]

        # 4) to Tensor
        x = torch.tensor(feat_vals, dtype=torch.float32)

        return x
