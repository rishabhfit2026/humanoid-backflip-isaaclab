# Humanoid Locomotion

Fork of [mujocolab/mjlab](https://github.com/mujocolab/mjlab) for the Humanoid bipedal robot.

![Humanoid in mjlab](docs/static/humanoid.png) 

---

## Demo

**Sim2Sim in MuJoCo** - Multi-directional velocity tracking:

!["Humanoid Sim2Sim Demo](docs/static/humanoid_sim.gif) 


---


## Quick Start

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and run
git clone https://github.com/Sentient-X/humanoid-mjlab.git
cd humanoid-mjlab
uv sync
```

### Train Humanoid Velocity Policy

```bash
uv run train Mjlab-Velocity-Flat-Humanoid --env.scene.num-envs 4096
```

### Evaluate Policy

```bash
uv run play Mjlab-Velocity-Flat-Humanoid --wandb-run-path your-org/mjlab/run-id
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Based on [mjlab](https://github.com/mujocolab/mjlab) by MuJoCo Lab.
