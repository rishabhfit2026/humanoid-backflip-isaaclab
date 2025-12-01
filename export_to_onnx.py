"""Export RSL-RL checkpoint to ONNX format."""

import torch
import torch.nn as nn


class PolicyForExport(nn.Module):
    """Policy network with normalization baked in for ONNX export."""
    def __init__(self, num_obs=45, num_actions=12):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_actions),
        )
        self.register_buffer('obs_mean', torch.zeros(1, num_obs))
        self.register_buffer('obs_std', torch.ones(1, num_obs))

    def forward(self, obs):
        obs_normalized = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return self.actor(obs_normalized)


def export_to_onnx(pt_path: str, onnx_path: str, num_obs=45, num_actions=12):
    """Export a PyTorch checkpoint to ONNX format.

    Args:
        pt_path: Path to the .pt checkpoint file
        onnx_path: Output path for the .onnx file
        num_obs: Number of observations (default: 45)
        num_actions: Number of actions (default: 12)
    """
    # Load checkpoint
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Create policy network
    policy = PolicyForExport(num_obs, num_actions)

    # Load actor weights
    actor_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('actor.') and not key.startswith('actor_obs'):
            actor_state_dict[key.replace('actor.', '')] = value
    policy.actor.load_state_dict(actor_state_dict)

    # Load normalizer weights if present
    if 'actor_obs_normalizer._mean' in state_dict:
        policy.obs_mean = state_dict['actor_obs_normalizer._mean']
        policy.obs_std = state_dict['actor_obs_normalizer._std']
        print(f"Loaded observation normalizer (mean shape: {policy.obs_mean.shape})")

    policy.eval()

    # Export to ONNX
    dummy_input = torch.zeros(1, num_obs)
    torch.onnx.export(
        policy,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['obs'],
        output_names=['actions'],
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'actions': {0: 'batch_size'}
        },
        dynamo=False,
    )
    print(f"Exported ONNX model to: {onnx_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Default paths
        pt_path = "/home/ishneet/Desktop/asimov-mjlab/logs/rsl_rl/asimov_toe_velocity/2025-12-01_15-59-05/model_2150.pt"
        onnx_path = pt_path.replace('.pt', '.onnx')
    else:
        pt_path = sys.argv[1]
        onnx_path = sys.argv[2] if len(sys.argv) > 2 else pt_path.replace('.pt', '.onnx')

    print(f"Converting: {pt_path}")
    print(f"Output: {onnx_path}")
    export_to_onnx(pt_path, onnx_path)
