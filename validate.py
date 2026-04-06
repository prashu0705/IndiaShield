"""
IndiaShield-v1 Validation Script
Confirms all OpenEnv spec requirements are met.
"""
from indiashield.env import IndiaShieldEnv, Action
from openenv.env import Env

def validate():
    print("Running IndiaShield-v1 validation...")
    print()

    env = IndiaShieldEnv("task1")
    assert isinstance(env, Env), "FAIL: Does not inherit from Env"
    print("✓ Inherits from openenv.env.Env")

    obs = env.reset()
    assert obs is not None, "FAIL: reset() returned None"
    print("✓ reset() returns observation")

    action = Action(type="identify_spreader")
    result = env.step(action)
    assert len(result) == 4, "FAIL: step() must return 4 values"
    obs, reward, done, info = result
    print("✓ step() returns (observation, reward, done, info)")

    assert isinstance(reward, float), "FAIL: reward must be float"
    assert -1.0 <= reward <= 1.0, f"FAIL: reward {reward} out of range"
    print(f"✓ reward is float in [-1.0, 1.0]: {reward}")

    assert isinstance(done, bool), "FAIL: done must be bool"
    print("✓ done is bool")

    state = env.state()
    assert isinstance(state, dict), "FAIL: state() must return dict"
    print("✓ state() returns dict")

    from indiashield.tasks import get_all_tasks
    tasks = get_all_tasks()
    assert len(tasks) == 5, f"FAIL: expected 5 tasks, got {len(tasks)}"
    print(f"✓ All 5 tasks load correctly")

    from indiashield.graders import grade
    for task in tasks:
        test_env = IndiaShieldEnv(task.task_id)
        test_env.reset()
        net_state = test_env.network.get_state()
        mod_state = test_env.model.get_state()
        result = grade(task.task_id, net_state, mod_state, task)
        assert 0.0 <= result.final_score <= 1.0, f"FAIL: {task.task_id} score out of range"
        print(f"✓ {task.task_id} grader score in [0.0, 1.0]: {result.final_score}")

    env2 = IndiaShieldEnv("task1")
    env2.reset()
    steps = 0
    while not env2.done and steps < 20:
        obs, reward, done, info = env2.step(Action(type="noop"))
        steps += 1
    assert env2.done, "FAIL: episode never terminated"
    print(f"✓ Episode terminates correctly after {steps} steps")

    assert env.name == "IndiaShield-v1", "FAIL: name not set"
    assert env.state_space is not None, "FAIL: state_space not set"
    assert env.action_space is not None, "FAIL: action_space not set"
    print("✓ name, state_space, action_space all set")

    print()
    print("=" * 40)
    print("ALL CHECKS PASSED — IndiaShield-v1 is valid")
    print("=" * 40)

if __name__ == "__main__":
    validate()
