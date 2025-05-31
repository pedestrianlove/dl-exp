import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

import os

from calories import CaloriesDataset, CaloriesTest
import polars as pl

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class Trainer:
    # Class members
    gpu_id: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    epochs_run: int
    batch_size: int

    # Model data
    criterion: torch.nn.Module
    train_sampler: DistributedSampler
    val_sampler: DistributedSampler
    train_loader: DataLoader
    val_loader: DataLoader
    test_sampler: DistributedSampler
    test_loader: DataLoader



    def __init__(
        self,
        gpu_id: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int
    ) -> None:
        # Prepare model for training
        self.gpu_id = int(gpu_id)
        self.model = DDP(model.to(self.gpu_id), device_ids=[self.gpu_id])
        self.optimizer = optimizer
        self.epochs_run = 0
        self.criterion = torch.nn.MSELoss().to(self.gpu_id)
        self.batch_size = batch_size

        # Process data
        dataset = CaloriesDataset('train.csv')
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size   = total_size - train_size
        training_set, validation_set = random_split(dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))
        self.train_sampler = DistributedSampler(
            training_set,
            num_replicas=torch.distributed.get_world_size(),
            rank=self.gpu_id,
            shuffle=True
        )
        self.val_sampler = DistributedSampler(
            validation_set,
            num_replicas=torch.distributed.get_world_size(),
            rank=self.gpu_id,
            shuffle=False
        )
        self.train_loader = DataLoader(training_set, batch_size=batch_size, sampler=self.train_sampler, num_workers=4)
        self.val_loader = DataLoader(validation_set, batch_size=batch_size, sampler=self.val_sampler, num_workers=4)


    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch, num_epochs=max_epochs)
    def _run_epoch(self, epoch: int, num_epochs: int = 10):
        self.model.train()
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.gpu_id), labels.to(self.gpu_id)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), labels).to(self.gpu_id)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.gpu_id), labels.to(self.gpu_id)
                outputs = self.model(inputs)
                val_loss += self.criterion(outputs.squeeze(), labels).item()
            val_loss /= len(self.val_loader)

        if self.gpu_id == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

    def generate_prediction(self, data_path: str):
        # Load the test dataset
        test_dataset = CaloriesTest(data_path)
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        # Inference
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs in self.test_loader:
                inputs = inputs.to(self.gpu_id)
                outputs = self.model(inputs).squeeze().cpu()
                predictions.append(outputs.numpy())

        # Save predictions to a CSV file
        all_predictions = np.concatenate(predictions, axis=0)
        predictions_df = pl.DataFrame({
            'id': test_dataset.rdata['id'],
            'Calories': all_predictions.tolist()
        })
        predictions_df.write_csv('predictions.csv')



def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

def ddp_cleanup():
    destroy_process_group()
