"""Test that all environments produce deterministic trajectories when seeded.

For every registered environment, four scenarios are verified:
  A. env1 freshly created, reset(seed=S) → trajectory
  B. env2 freshly created, reset(seed=S) → same trajectory
  C. env1 after running an episode, reset(seed=S) → same trajectory
  D. env2 after running an episode, reset(seed=S) → same trajectory

Additionally, a seed-switch test verifies that resetting with a *different*
seed fully erases prior state:
  env1: reset(seed1) → run → reset(seed2) → trajectory X
  env2 (fresh): reset(seed2) → trajectory Y
  X must equal Y.
"""

import gymnasium as gym
import numpy as np
import pytest

import bullet_safety_gym  # noqa: F401 – registers envs

# Collect all registered Safety env ids.
ALL_ENV_IDS = sorted(
    spec.id
    for spec in gym.envs.registry.values()
    if 'Safety' in spec.id
)

# Multiple seeds are exercised so that flaky physics-engine non-determinism
# (which only manifests for certain seed/state combinations) is more likely to
# be caught locally rather than appearing only on CI.
SEEDS = [0, 7, 42, 123, 2024]
NUM_STEPS = 15
# Separate fixed seed for action generation — must be independent of the env
# seed so that the same action sequence is replayed across all scenarios.
ACTION_SEED = 0


def collect_trajectory(env, seed, num_steps=NUM_STEPS):
    """Reset *env* with *seed* and step with randomly sampled actions.

    Actions are drawn from a fixed RNG (ACTION_SEED) that is completely
    independent of the environment seed, guaranteeing the same action
    sequence is used in every call regardless of *seed*.
    """
    action_rng = np.random.default_rng(ACTION_SEED)
    obs, _ = env.reset(seed=seed)
    observations = [obs.copy()]
    rewards = []
    for _ in range(num_steps):
        action = action_rng.uniform(
            env.action_space.low, env.action_space.high
        ).astype(env.action_space.dtype)
        obs, r, terminated, truncated, _info = env.step(action)
        observations.append(obs.copy())
        rewards.append(r)
        if terminated or truncated:
            break
    return observations, rewards


def assert_trajectories_equal(traj_a, traj_b, label=""):
    """Assert two (obs_list, reward_list) tuples are identical."""
    obs_a, rew_a = traj_a
    obs_b, rew_b = traj_b
    assert len(obs_a) == len(obs_b), (
        f"{label}: trajectory lengths differ ({len(obs_a)} vs {len(obs_b)})"
    )
    for step, (oa, ob) in enumerate(zip(obs_a, obs_b)):
        np.testing.assert_array_equal(
            oa, ob,
            err_msg=f"{label}: observations differ at step {step}",
        )
    for step, (ra, rb) in enumerate(zip(rew_a, rew_b)):
        assert ra == rb, (
            f"{label}: rewards differ at step {step}: {ra} vs {rb}"
        )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
def test_seed_determinism_across_instances_and_resets(env_id, seed):
    """Same seed → identical trajectory, across fresh instances and re-resets."""
    # --- Scenario A: fresh env1 ---
    env1 = gym.make(env_id)
    traj_a = collect_trajectory(env1, seed=seed)

    # --- Scenario B: fresh env2 ---
    env2 = gym.make(env_id)
    traj_b = collect_trajectory(env2, seed=seed)
    assert_trajectories_equal(traj_a, traj_b, label="A-vs-B (cross-instance)")

    # Run both envs for a while to change internal state
    for _env in (env1, env2):
        _env.reset()
        for _ in range(30):
            _env.step(_env.action_space.sample())

    # --- Scenario C: env1 re-reset with same seed ---
    traj_c = collect_trajectory(env1, seed=seed)
    assert_trajectories_equal(traj_a, traj_c, label="A-vs-C (env1 re-reset)")

    # --- Scenario D: env2 re-reset with same seed ---
    traj_d = collect_trajectory(env2, seed=seed)
    assert_trajectories_equal(traj_a, traj_d, label="A-vs-D (env2 re-reset)")

    env1.close()
    env2.close()


# Pairs of (dirty_seed, clean_seed): we run with dirty_seed first, then
# reset with clean_seed and compare against a fresh env seeded with clean_seed.
SEED_SWITCH_PAIRS = [
    (0, 99),
    (42, 99),
    (7, 13),
    (100, 200),
    (2024, 1),
]


@pytest.mark.parametrize("seed1, seed2", SEED_SWITCH_PAIRS)
@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
def test_seed_switch_determinism(env_id, seed1, seed2):
    """Switching seeds must erase all prior state.

    env1: reset(seed1) → run → reset(seed2) → trajectory X
    env2: reset(seed2) → trajectory Y
    X must equal Y.
    """
    # env1: run with seed1, then switch to seed2
    env1 = gym.make(env_id)
    env1.reset(seed=seed1)
    for _ in range(30):
        env1.step(env1.action_space.sample())
    traj_x = collect_trajectory(env1, seed=seed2)

    # env2: fresh start with seed2
    env2 = gym.make(env_id)
    traj_y = collect_trajectory(env2, seed=seed2)

    assert_trajectories_equal(traj_x, traj_y, label="seed-switch")

    env1.close()
    env2.close()
