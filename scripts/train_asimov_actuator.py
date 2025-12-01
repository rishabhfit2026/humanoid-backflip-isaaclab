"""Train learned MLP actuator network for Asimov robot from CAN bus data.

Usage:
    python scripts/train_asimov_actuator.py --csv data.csv --output asimov_actuator.pt

The network learns to predict torque from joint state history:
    Input: [pos_error[t], pos_error[t-1], pos_error[t-2], vel[t], vel[t-1], vel[t-2]]
    Output: torque

This follows the walk-these-ways actuator network architecture (Hwangbo et al. 2019).

References:
- Walk-these-ways: https://github.com/Improbable-AI/walk-these-ways
- Hwangbo et al. (2019): https://arxiv.org/abs/1901.08652
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Motor torque constants (Nm/A) for each motor ID.
# Data from Asimov hardware specs.
MOTOR_KT = {
    1: 2.8,   # L_Hip_Pitch
    2: 7.26,  # L_Hip_Roll
    3: 5.7,   # L_Hip_Yaw
    4: 2.53,  # L_Knee
    5: 1.8,   # L_Ankle_Pitch (excluded - tendon A/B control)
    6: 1.8,   # L_Ankle_Roll (excluded - tendon A/B control)
    7: 2.8,   # R_Hip_Pitch
    8: 7.26,  # R_Hip_Roll
    9: 5.7,   # R_Hip_Yaw
    10: 2.53, # R_Knee
    11: 1.8,  # R_Ankle_Pitch (excluded - tendon A/B control)
    12: 1.8,  # R_Ankle_Roll (excluded - tendon A/B control)
}

# Joint name mapping for nicer plotting.
JOINT_NAMES = {
    1: "L_Hip_Pitch",
    2: "L_Hip_Roll",
    3: "L_Hip_Yaw",
    4: "L_Knee",
    5: "L_Ankle_Pitch",
    6: "L_Ankle_Roll",
    7: "R_Hip_Pitch",
    8: "R_Hip_Roll",
    9: "R_Hip_Yaw",
    10: "R_Knee",
    11: "R_Ankle_Pitch",
    12: "R_Ankle_Roll",
}

# Motor IDs to train on (hip and knee only).
# Ankles (5,6,11,12) are excluded because they use tendon A/B control in the XML.
TRAIN_MOTOR_IDS = [1, 2, 3, 4, 7, 8, 9, 10]


class ActuatorDataset(Dataset):
    """Dataset for actuator network training."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        x = np.concatenate([s["pos_error_history"], s["vel_history"]])
        return torch.from_numpy(x), torch.tensor(s["torque"], dtype=torch.float32)


class SoftsignActivation(nn.Module):
    """Softsign activation: x / (1 + |x|)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softsign(x)


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    out_dim: int,
) -> nn.Sequential:
    """Build MLP with softsign activation (matches walk-these-ways)."""
    layers = [nn.Linear(in_dim, hidden_dim), SoftsignActivation()]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), SoftsignActivation()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def load_and_preprocess_data(
    csv_path: str,
    history_length: int = 3,
    prediction_offset: int = 2,
    min_velocity: float = 0.0,
    train_split: float = 0.8,
) -> tuple[Dataset, Dataset]:
    """Load CAN bus CSV and prepare training/validation datasets.

    Args:
        csv_path: Path to CSV with columns:
            timestamp,motor_id,motor_name,cmd_pos,fb_pos,fb_vel,fb_cur,fb_temp,error_code
        history_length: Number of timesteps of history to use.
        prediction_offset: Predict torque N steps ahead (walk-these-ways: 2).
        min_velocity: Only include samples where |vel| > this (filters standing).
        train_split: Fraction of data for training.

    Returns:
        train_dataset, val_dataset
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter to hip/knee motors only (exclude ankles which use tendon A/B control).
    df = df[df["motor_id"].isin(TRAIN_MOTOR_IDS)].copy()

    # Compute torque from current: τ = I × kt
    df["torque"] = df.apply(lambda row: row["fb_cur"] * MOTOR_KT[row["motor_id"]], axis=1)

    # Compute position error: error = cmd_pos - fb_pos (target - current).
    # This matches mjlab's LearnedMlpActuator which computes (target - current),
    # so we use pos_scale=1.0 in the config.
    df["pos_error"] = df["cmd_pos"] - df["fb_pos"]

    # Sort by motor_id and timestamp.
    df = df.sort_values(["motor_id", "timestamp"])

    print(f"Total samples: {len(df)}")
    print(f"Samples per joint: ~{len(df) // len(TRAIN_MOTOR_IDS)}")
    print(f"Training on motors: {TRAIN_MOTOR_IDS} (hip/knee only, ankles excluded)")

    # Build training samples for each joint.
    all_samples = []

    for motor_id in TRAIN_MOTOR_IDS:
        motor_df = df[df["motor_id"] == motor_id].reset_index(drop=True)

        pos_errors = motor_df["pos_error"].values.astype(np.float32)
        velocities = motor_df["fb_vel"].values.astype(np.float32)
        torques = motor_df["torque"].values.astype(np.float32)

        n_samples = len(motor_df)

        # Walk-these-ways windowing:
        # History includes current timestep: [t-history_length+1, ..., t]
        # Target is torque at t + prediction_offset
        for i in range(history_length - 1, n_samples - prediction_offset):
            # Check velocity filter (use max velocity in window).
            vel_window = np.abs(velocities[i - history_length + 1 : i + 1])
            if min_velocity > 0 and vel_window.max() < min_velocity:
                continue

            # Input: [pos_error[t-2:t], vel[t-2:t]] (history_length=3)
            pos_history = pos_errors[i - history_length + 1 : i + 1]
            vel_history = velocities[i - history_length + 1 : i + 1]

            # Output: torque at t + prediction_offset
            torque = torques[i + prediction_offset]

            all_samples.append({
                "motor_id": motor_id,
                "pos_error_history": pos_history,
                "vel_history": vel_history,
                "torque": torque,
            })

    # Shuffle all samples to break temporal correlation (critical for generalization).
    random.seed(42)
    random.shuffle(all_samples)

    # Train/val split.
    n_train = int(len(all_samples) * train_split)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]

    print(f"Total samples after filtering: {len(all_samples)}")
    if min_velocity > 0:
        print(f"  (filtered to |vel| > {min_velocity} rad/s)")
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    return ActuatorDataset(train_samples), ActuatorDataset(val_samples)


def train_actuator_network(
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_path: str,
    history_length: int = 3,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 8e-4,
    device: str = "cuda:0",
) -> nn.Module:
    """Train the actuator network.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        output_path: Path to save TorchScript model.
        history_length: Number of history timesteps (for input dim calculation).
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: Training device.

    Returns:
        Trained model.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Build model: MLP with 32 hidden units, 2 layers, softsign activation.
    # Input: 2 * history_length features (pos_error + vel), Output: 1 torque.
    input_dim = 2 * history_length
    model = build_mlp(in_dim=input_dim, hidden_dim=32, num_layers=2, out_dim=1)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    print(f"\nTraining on {device} for {epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Model: MLP({input_dim} -> 32 -> 32 -> 1) with softsign")

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        # Training.
        model.train()
        train_loss = 0.0
        train_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(x).squeeze(-1)
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches

        # Validation.
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                y_pred = model(x).squeeze(-1)
                loss = F.mse_loss(y_pred, y)
                mae = (y_pred - y).abs().mean()

                val_loss += loss.item()
                val_mae += mae.item()
                val_batches += 1

        val_loss /= val_batches
        val_mae /= val_batches

        # Update LR scheduler.
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save best model.
            model_cpu = model.cpu()
            model_scripted = torch.jit.script(model_cpu)
            model_scripted.save(output_path)
            model = model.to(device)

        if epoch % 10 == 0 or epoch == epochs - 1:
            train_rmse = np.sqrt(train_loss)
            val_rmse = np.sqrt(val_loss)
            print(
                f"Epoch {epoch:3d} | "
                f"Train RMSE: {train_rmse:.4f} Nm | "
                f"Val RMSE: {val_rmse:.4f} Nm | "
                f"Val MAE: {val_mae:.4f} Nm"
            )

    print(f"\nBest validation RMSE: {np.sqrt(best_val_loss):.4f} Nm (epoch {best_epoch})")
    print(f"Model saved to: {output_path}")

    # Reload best model.
    model = torch.jit.load(output_path, map_location=device)
    return model


def evaluate_and_plot(
    model: nn.Module,
    csv_path: str,
    history_length: int = 3,
    plot_length: int = 500,
    min_velocity: float = 0.0,
    device: str = "cpu",
) -> None:
    """Evaluate model and plot predictions vs ground truth."""
    df = pd.read_csv(csv_path)
    df = df[df["motor_id"].isin(TRAIN_MOTOR_IDS)].copy()
    df["torque"] = df.apply(lambda row: row["fb_cur"] * MOTOR_KT[row["motor_id"]], axis=1)
    df["pos_error"] = df["cmd_pos"] - df["fb_pos"]  # target - current (matches training)
    df = df.sort_values(["motor_id", "timestamp"])

    model = model.to(device)
    model.eval()

    # 8 motors (4 per leg, hip/knee only)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    step = 2
    all_errors = []

    for motor_idx, motor_id in enumerate(TRAIN_MOTOR_IDS):
        motor_df = df[df["motor_id"] == motor_id].reset_index(drop=True)

        pos_errors = motor_df["pos_error"].values.astype(np.float32)
        velocities = motor_df["fb_vel"].values.astype(np.float32)
        torques = motor_df["torque"].values

        # Compute predictions using same windowing as training.
        preds = []
        actual_list = []
        sample_indices = []

        for i in range(history_length - 1, len(motor_df) - step):
            # Apply same velocity filter as training.
            vel_window = np.abs(velocities[i - history_length + 1 : i + 1])
            if min_velocity > 0 and vel_window.max() < min_velocity:
                continue

            # Same windowing as training: [t-history_length+1, ..., t]
            pos_history = pos_errors[i - history_length + 1 : i + 1]
            vel_history = velocities[i - history_length + 1 : i + 1]
            x = np.concatenate([pos_history, vel_history])
            x = torch.tensor(x).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(x).squeeze().item()
            preds.append(pred)
            actual_list.append(torques[i + step])
            sample_indices.append(i)

            if len(preds) >= plot_length:
                break

        preds = np.array(preds)
        actual = np.array(actual_list)
        errors = np.abs(preds - actual)
        all_errors.extend(errors)

        # Use sample index for x-axis (since samples may not be contiguous).
        t = np.arange(len(preds))

        ax = axes[motor_idx]
        ax.plot(t, actual, label="Measured", alpha=0.7, linewidth=1)
        ax.plot(t, preds, label="Predicted", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_title(f"{JOINT_NAMES[motor_id]} (MAE: {errors.mean():.2f} Nm)")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Torque (Nm)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    overall_mae = np.mean(all_errors)
    filter_str = f" (|vel| > {min_velocity} rad/s)" if min_velocity > 0 else ""
    plt.suptitle(f"Asimov Actuator Network{filter_str} - Overall MAE: {overall_mae:.2f} Nm", fontsize=14)
    plt.tight_layout()
    plt.savefig("asimov_actuator_predictions.png", dpi=150)
    plt.show()
    print(f"Plot saved to: asimov_actuator_predictions.png")
    print(f"Overall MAE on plotted samples: {overall_mae:.2f} Nm")


def main():
    parser = argparse.ArgumentParser(description="Train Asimov actuator network")
    parser.add_argument("--csv", type=str, default="data.csv", help="Path to CAN bus CSV")
    parser.add_argument(
        "--output",
        type=str,
        default="asimov_actuator.pt",
        help="Output path for TorchScript model",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--history-length", type=int, default=3, help="History timesteps")
    parser.add_argument("--prediction-offset", type=int, default=2,
                        help="Predict torque N steps ahead (captures actuator delay)")
    parser.add_argument("--min-velocity", type=float, default=0.0,
                        help="Only train on samples where |vel| > this (filters standing)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    parser.add_argument("--plot", action="store_true", help="Plot predictions after training")
    parser.add_argument("--plot-length", type=int, default=500, help="Number of samples to plot")
    args = parser.parse_args()

    print("=" * 60)
    print("Walk-These-Ways Actuator Network Training")
    print("=" * 60)
    print(f"History length: {args.history_length}")
    print(f"Prediction offset: {args.prediction_offset} (predicts torque {args.prediction_offset} steps ahead)")
    print(f"Min velocity filter: {args.min_velocity} rad/s")
    print("=" * 60)

    # Load and preprocess data.
    train_dataset, val_dataset = load_and_preprocess_data(
        args.csv,
        history_length=args.history_length,
        prediction_offset=args.prediction_offset,
        min_velocity=args.min_velocity,
    )

    # Train network.
    model = train_actuator_network(
        train_dataset,
        val_dataset,
        args.output,
        history_length=args.history_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    # Optionally plot predictions.
    if args.plot:
        evaluate_and_plot(
            model,
            args.csv,
            history_length=args.history_length,
            plot_length=args.plot_length,
            min_velocity=args.min_velocity,
            device="cpu",
        )

    print("\nDone! To use the trained model in mjlab:")
    print(f"""
from mjlab.asset_zoo.robots.asimov.asimov_toe_constants import (
    get_asimov_robot_cfg_learned,
)

# The learned actuator config is already defined in asimov_toe_constants.py.
# It covers hip_pitch, hip_roll, hip_yaw, knee joints (8 total).
# Ankle joints use tendon A/B control defined in the XML.

# Copy the trained model to the expected location:
#   cp {args.output} src/mjlab/asset_zoo/robots/asimov/assets/asimov_actuator.pt

# Then use in your environment:
robot_cfg = get_asimov_robot_cfg_learned()
""")


if __name__ == "__main__":
    main()
