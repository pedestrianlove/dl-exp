import os
import torch

from trainer import Trainer, ddp_setup, ddp_cleanup
from calories import CaloriesPrediction

# Setup DDP
ddp_setup()

## Initialize the model
model = CaloriesPrediction()
calories_trainer = Trainer(
    gpu_id=os.environ["LOCAL_RANK"],
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=0.0005),
    batch_size=3200,  # Adjust batch size as needed
)

## Train the model
calories_trainer.train(max_epochs=600)  # Adjust max_epochs as needed

## Generate predictions
if os.environ.get("LOCAL_RANK") == "0":
    calories_trainer.generate_prediction(data_path='test.csv')

# Destroy DDP
ddp_cleanup()
