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
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    batch_size=320,  # Adjust batch size as needed
)

## Train the model
calories_trainer.train(max_epochs=20)  # Adjust max_epochs as needed

# Destroy DDP
ddp_cleanup()
