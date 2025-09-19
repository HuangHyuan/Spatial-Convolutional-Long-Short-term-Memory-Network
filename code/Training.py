import torch
import torch.utils.data as Data
import logging
import os
from typing import Dict, Any, Tuple
from model import generate_SCSL as SCSL
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train the SCSL model using atmospheric and surface data.

    Args:
        config (dict): Configuration dictionary containing:
            - x_atm, x_surf, y_atm, y_surf: Training tensors
            - x_atm_val, x_surf_val, y_atm_val, y_surf_val: Validation tensors
            - batch (int): Batch size
            - learning_rate (float): Learning rate for optimizer
            - criterion: Loss function
            - num_epochs (int): Number of training epochs
            - model_save_path (str, optional): Path to save the best model (default: './train_model/best_SCSL_another.pth')

    Returns:
        dict: Contains training/validation loss history and the trained model.
    """
    # --- 1. Load and prepare datasets ---
    logger.info("Preparing training and validation datasets...")

    try:
        train_dataset = Data.TensorDataset(
            config['x_atm'],
            config['x_surf'],
            config['y_atm'],
            config['y_surf']
        )
        val_dataset = Data.TensorDataset(
            config['x_atm_val'],
            config['x_surf_val'],
            config['y_atm_val'],
            config['y_surf_val']
        )
    except Exception as e:
        logger.error("Failed to create datasets.", exc_info=True)
        raise e

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=config['batch'],
        shuffle=True,
        num_workers=config.get('num_workers', 1),
        pin_memory=torch.cuda.is_available()
    )
    val_loader = Data.DataLoader(
        dataset=val_dataset,
        batch_size=config['batch'],
        shuffle=False,
        num_workers=config.get('num_workers', 1),
        pin_memory=torch.cuda.is_available()
    )

    # --- 2. Initialize model and optimizer ---
    logger.info("Initializing model and optimizer...")
    model = SCSL()  # Ensure SCSL is defined or imported
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adadelta(
        model.parameters(),
        lr=config['learning_rate'],
        rho=0.9,
        eps=1e-06,
        weight_decay=0
    )
    criterion = config['criterion']

    # --- 3. Training setup ---
    num_epochs = config['num_epochs']
    losses = []
    val_losses = []
    loss_atm = []
    loss_surf = []
    best_val_loss = float('inf')
    model_save_path = config.get('model_save_path', './train_model/best_SCSL_another.pth')

    # Create directory if not exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    logger.info(f"Starting training for {num_epochs} epochs on {device}...")

    # --- 4. Training loop ---
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_loss_atm = 0.0
        train_loss_surf = 0.0
        total_samples = 0

        for step, (x_atm_batch, x_surf_batch, y_atm_batch, y_surf_batch) in enumerate(train_loader):
            x_atm_batch = x_atm_batch.to(device, non_blocking=True)
            x_surf_batch = x_surf_batch.to(device, non_blocking=True)
            y_atm_batch = y_atm_batch.to(device, non_blocking=True)
            y_surf_batch = y_surf_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            y_pred_atm, y_pred_surf = model(x_atm_batch, x_surf_batch)

            # Compute losses
            loss_atm_step = criterion(y_pred_atm, y_atm_batch)
            loss_surf_step = criterion(y_pred_surf, y_surf_batch)
            total_loss = loss_atm_step + 0.5 * loss_surf_step

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            batch_size = x_atm_batch.size(0)
            train_loss += total_loss.item() * batch_size
            train_loss_atm += loss_atm_step.item() * batch_size
            train_loss_surf += loss_surf_step.item() * batch_size
            total_samples += batch_size

        # Average training losses
        avg_train_loss = train_loss / total_samples
        avg_train_loss_atm = train_loss_atm / total_samples
        avg_train_loss_surf = train_loss_surf / total_samples

        losses.append(avg_train_loss)
        loss_atm.append(avg_train_loss_atm)
        loss_surf.append(avg_train_loss_surf)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_total_samples = 0

        with torch.no_grad():
            for x_atm_val_batch, x_surf_val_batch, y_atm_val_batch, y_surf_val_batch in val_loader:
                x_atm_val_batch = x_atm_val_batch.to(device, non_blocking=True)
                x_surf_val_batch = x_surf_val_batch.to(device, non_blocking=True)
                y_atm_val_batch = y_atm_val_batch.to(device, non_blocking=True)
                y_surf_val_batch = y_surf_val_batch.to(device, non_blocking=True)

                y_pred_atm, y_pred_surf = model(x_atm_val_batch, x_surf_val_batch)
                val_loss_atm = criterion(y_pred_atm, y_atm_val_batch)
                val_loss_surf = criterion(y_pred_surf, y_surf_val_batch)
                total_val_loss = val_loss_atm + 0.5 * val_loss_surf

                batch_size = x_atm_val_batch.size(0)
                val_loss += total_val_loss.item() * batch_size
                val_total_samples += batch_size

        avg_val_loss = val_loss / val_total_samples
        val_losses.append(avg_val_loss)

        # --- Model checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)  # Save only state_dict is safer
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

        # --- Logging ---
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} "
            f"(Atm: {avg_train_loss_atm:.4f}, Surf: {avg_train_loss_surf:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Best Val Loss: {best_val_loss:.4f}"
        )

    # --- 5. Final result ---
    logger.info("Training completed.")
    return {
        "train_loss": losses,
        "train_atm_loss": loss_atm,
        "train_sur_loss": loss_surf,
        "val_loss": val_losses,
        "model": model,  # Note: model is on GPU; consider moving to CPU before returning if needed
        "best_val_loss": best_val_loss
    }