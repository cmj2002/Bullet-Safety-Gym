# Bullet-Safety-Gym (Fixed Fork)

This is a **personal maintenance fork** of [Bullet-Safety-Gym](https://github.com/liuzuxin/Bullet-Safety-Gym), which is no longer actively maintained upstream.

The original README is preserved in [OLD_README.md](OLD_README.md).

## What's Changed

### Seed Determinism

The original codebase used `np.random.uniform()`, `np.random.randint()`, and `random.uniform()` (module-level global RNG state) throughout agents, tasks, worlds, and obstacles. This made it impossible to reproduce episodes by setting a seed.

**Fix:** Introduced a shared `np.random.Generator` instance (`rng`) that is created in `EnvironmentBuilder` and propagated to all components (`Agent`, `Task`, `World`, `Obstacle`). Both `env.seed()` and `env.reset(seed=...)` properly recreate and propagate the RNG.

### Deterministic Obstacle Movement

Moving obstacles used `time.time()` (wall-clock time) to compute their trajectories, making their positions non-deterministic.

**Fix:** Replaced `time.time()` with a simulation-step counter (`_movement_step`) so obstacle movement is tied to physics steps, not real time.

### Agent State Reset

Several agents did not fully reset their internal state between episodes, causing state leakage:

- **Ball**: `last_taken_action` was not reset.
- **RaceCar**: Motor control commands (velocity/position targets) carried over from the previous episode.
- **MJCFAgent / Ant**: `feet_collision_reward`, `joints_at_limit_reward`, `action_reward`, and `last_taken_action` were not reset.
- **Drone**: Rotor ground contact state, ground collision penalty, and last action were not reset.

**Fix:** Added proper state resets in each agent's `specific_reset()` method.

### `init_xyz` Mutation Bug

`ReachGoalTask` and `RunTask` directly mutated `agent.init_xyz` when setting new episode positions (`agent_pos = self.agent.init_xyz`, then `agent_pos[:2] = ...`). This permanently altered the agent's initial position template.

**Fix:** Use `.copy()` before modifying: `agent_pos = self.agent.init_xyz.copy()`.

### Obstacle Position Generation Bug

In `ReachGoalTask.specific_reset()`, obstacle initial positions were generated based on `self.agent.get_position()` and `self.goal.get_position()` — i.e., the **previous** episode's positions, since `set_position` hadn't been called yet.

**Fix:** Pass the newly computed local `agent_pos` and `goal_pos` variables instead.

### PyBullet Output Suppression

The original `RedirectStream` relied on `sys.stdout` / `sys.stderr` Python objects, which break under pytest's fd-capture. PyBullet's C code writes directly to fd 1/fd 2.

**Fix:** Rewrote `RedirectStream` to operate on raw file descriptor numbers. Added an `atexit` handler to suppress PyBullet's `argv[0]=` exit noise.

### `close()` Double-Disconnect Prevention

Calling `env.close()` could trigger a second `disconnect()` in `BulletClient.__del__`, producing noisy errors.

**Fix:** Set `bc._client = -1` after disconnect to prevent the destructor from disconnecting again, and wrapped `disconnect()` in `RedirectStream(1)` to suppress output.

### New Tests

Added three tests in `tests/test_envs.py`:

- `test_seed_determinism` — Two episodes with the same seed produce identical trajectories.
- `test_reset_no_state_leak` — State from a previous episode does not leak after reset.
- `test_init_xyz_not_mutated` — Agent's `init_xyz` is not mutated across resets.

## Installation

```bash
pip install git+https://github.com/cmj2002/Bullet-Safety-Gym.git
```

## Quick Start

```python
import gymnasium as gym
import bullet_safety_gym

env = gym.make('SafetyBallReach-v0')
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## License

[MIT](LICENSE) (same as the original repository).
