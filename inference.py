"""
Inference Script — IndiaShield-v1
===================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import json
import re
from typing import Any, List, Optional
from indiashield.env import IndiaShieldEnv, Action

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_key")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
BENCHMARK = "indiashield-v1"
MAX_STEPS = 15
TEMPERATURE = 0.2
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5


def get_client() -> Optional[Any]:
    if OpenAI is None:
        return None

    try:
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[WARN] OpenAI client init failed: {str(exc)[:120]}", flush=True)
        return None


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = """You are an AI agent controlling IndiaShield — a system that:
1. Contains WhatsApp misinformation outbreaks in India
2. Compresses a multilingual AI classifier for deployment on cheap phones

Each turn choose exactly one action from this list:

NETWORK ACTIONS:
- {"type": "identify_spreader"} — find the biggest spreader
- {"type": "intercept", "node_id": <int>} — block one infected node
- {"type": "quarantine", "group_id": <int>} — block entire group
- {"type": "add_forward_label", "node_id": <int>} — mark message as forwarded

MODEL ACTIONS:
- {"type": "quantize", "precision": "int8"} — shrink 55%, small accuracy loss
- {"type": "quantize", "precision": "int4"} — shrink 72%, large accuracy loss
- {"type": "prune", "target_layer": "ffn", "percentage": 30} — safe pruning
- {"type": "prune", "target_layer": "attention", "percentage": 30} — hurts Hindi
- {"type": "prune", "target_layer": "all", "percentage": 50} — aggressive
- {"type": "distill", "student_size": "small"} — best compression, takes 2 turns
- {"type": "distill", "student_size": "tiny"} — smallest model, takes 3 turns

DEPLOYMENT:
- {"type": "deploy", "node_id": <int>} — install model on phone (only if model is small enough)

STRATEGY:
- Compress model early so you can deploy later
- Intercept super spreaders first — they have most connections
- Balance network actions AND model compression every episode
- Deploy to as many nodes as possible once model is small enough

Respond with ONLY a JSON action object. Nothing else."""


def build_prompt(obs: dict, turn: int) -> str:
    infected_pct = round(obs["infected"] / obs["total_users"] * 100, 1)
    size_gap = round(obs["model_size_mb"] - obs["target_size_mb"], 1)
    return f"""TURN {turn}/{obs['max_turns']} — {obs['task_name']}

NETWORK STATUS:
- Infected: {obs['infected']}/{obs['total_users']} ({infected_pct}%)
- Sources found: {obs['sources_found']}/{obs['total_sources']}
- Super spreaders found: {obs['super_spreaders_found']}/{obs['total_super_spreaders']}

MODEL STATUS:
- Size: {obs['model_size_mb']}mb (need {obs['target_size_mb']}mb, gap: {size_gap}mb)
- Ready to deploy: {obs['model_ready_to_deploy']}
- Hindi accuracy: {obs['hindi_accuracy']}
- Tamil accuracy: {obs['tamil_accuracy']}
- Nodes protected: {obs['nodes_protected']}
- Distill turns remaining: {obs['distill_turns_remaining']}

LAST ACTION: {json.dumps(obs.get('last_action_result', {}))}

Reply with JSON action only."""


def parse_action(response_text: str) -> Action:
    response_text = response_text.strip()
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        return Action(type="noop")
    try:
        action_dict = json.loads(json_match.group())
        return Action(**action_dict)
    except Exception:
        return Action(type="noop")


def run_task(task_id: str) -> dict:
    env = IndiaShieldEnv(task_id=task_id)
    obs_obj = env.reset()
    obs = obs_obj.model_dump()
    client = get_client()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if env.done:
                break

            user_prompt = build_prompt(obs, step)
            messages.append({"role": "user", "content": user_prompt})

            error = None
            if client is None:
                response_text = '{"type": "noop"}'
                error = "OpenAI client unavailable"
                messages.append({"role": "assistant", "content": response_text})
            else:
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS
                    )
                    response_text = response.choices[0].message.content or '{"type": "noop"}'
                    messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    response_text = '{"type": "noop"}'
                    error = str(e)[:100]
                    messages.append({"role": "assistant", "content": response_text})

            action = parse_action(response_text)
            obs_obj, reward, done, info = env.step(action)
            obs = obs_obj.model_dump()

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action.type,
                reward=reward,
                done=done,
                error=error
            )

            if done:
                break

        result = env.grade()
        score = float(result.final_score)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "final_score": score,
        "passed": success,
        "total_reward": round(sum(rewards), 3),
        "breakdown": result.breakdown if 'result' in dir() else {},
        "feedback": result.feedback if 'result' in dir() else ""
    }


def main():
    task_ids = ["task1", "task2", "task3"]
    results = []

    for task_id in task_ids:
        result = run_task(task_id)
        results.append(result)

    avg = round(sum(r["final_score"] for r in results) / len(results), 3)

    with open("baseline_scores.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "results": results,
            "average_score": avg
        }, f, indent=2)

    print(f"\nAverage score: {avg}", flush=True)
    print("Scores saved to baseline_scores.json", flush=True)


if __name__ == "__main__":
    main()
