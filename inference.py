"""
Inference Script — IndiaShield-v1
===================================
Baseline agent that runs an LLM against the IndiaShield environment.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier to use
    HF_TOKEN       Your Hugging Face / API key
"""

import os
import json
import re
import time
from typing import Dict, Any, Optional
from openai import OpenAI
from indiashield.env import IndiaShieldEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_key")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
MAX_STEPS = 15
TEMPERATURE = 0.2
MAX_TOKENS = 300

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

SYSTEM_PROMPT = """You are an AI agent controlling IndiaShield — a system that:
1. Contains WhatsApp misinformation outbreaks in India
2. Compresses a multilingual AI classifier for deployment on cheap phones

Each turn you must choose exactly one action from this list:

NETWORK ACTIONS (fight the spread):
- {"type": "identify_spreader"} — find the biggest spreader (costs 1 turn)
- {"type": "intercept", "node_id": <int>} — block one infected node
- {"type": "quarantine", "group_id": <int>} — block entire group
- {"type": "add_forward_label", "node_id": <int>} — mark message as forwarded

MODEL ACTIONS (shrink the classifier):
- {"type": "quantize", "precision": "int8"} — shrink 55%, small accuracy loss
- {"type": "quantize", "precision": "int4"} — shrink 72%, large accuracy loss
- {"type": "prune", "target_layer": "ffn", "percentage": 30} — safe pruning
- {"type": "prune", "target_layer": "attention", "percentage": 30} — hurts Hindi
- {"type": "prune", "target_layer": "all", "percentage": 50} — aggressive
- {"type": "distill", "student_size": "small"} — best compression, takes 2 turns
- {"type": "distill", "student_size": "tiny"} — smallest model, takes 3 turns

DEPLOYMENT ACTION:
- {"type": "deploy", "node_id": <int>} — install model on one phone (only works if model is small enough)

STRATEGY TIPS:
- Balance network actions AND model compression every episode
- Compress model early so you can deploy later
- intercept super spreaders first — they have the most connections
- deploy to as many nodes as possible once model is small enough
- do NOT waste turns on noop

Respond with ONLY a JSON action object. Nothing else. No explanation."""


def build_prompt(obs: Dict[str, Any], turn: int) -> str:
    infected_pct = round(obs["infected"] / obs["total_users"] * 100, 1)
    size_gap = round(obs["model_size_mb"] - obs["target_size_mb"], 1)

    prompt = f"""TURN {turn}/{obs['max_turns']} — {obs['task_name']}

NETWORK STATUS:
- Infected: {obs['infected']}/{obs['total_users']} ({infected_pct}%)
- Blocked: {obs['blocked']}
- Sources found: {obs['sources_found']}/{obs['total_sources']}
- Super spreaders found: {obs['super_spreaders_found']}/{obs['total_super_spreaders']}

MODEL STATUS:
- Current size: {obs['model_size_mb']}mb (need to get to {obs['target_size_mb']}mb, gap: {size_gap}mb)
- Ready to deploy: {obs['model_ready_to_deploy']}
- Hindi accuracy: {obs['hindi_accuracy']}
- Tamil accuracy: {obs['tamil_accuracy']}
- Nodes protected: {obs['nodes_protected']}
- Distill turns remaining: {obs['distill_turns_remaining']}

LAST ACTION RESULT:
{json.dumps(obs.get('last_action_result', {}), indent=2)}

What is your next action? Reply with JSON only."""
    return prompt


def parse_action(response_text: str) -> Optional[Action]:
    response_text = response_text.strip()
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not json_match:
        return Action(type="noop")
    try:
        action_dict = json.loads(json_match.group())
        return Action(**action_dict)
    except Exception:
        return Action(type="noop")


def run_task(task_id: str) -> Dict[str, Any]:
    print(f"\n{'='*50}")
    print(f"Running task: {task_id}")
    print(f"{'='*50}")

    env = IndiaShieldEnv(task_id=task_id)
    obs_obj = env.reset()
    obs = obs_obj.model_dump()

    print(f"Story: {obs['task_story'][:80]}...")
    print(f"Users: {obs['total_users']} | "
          f"Infected: {obs['infected']} | "
          f"Model: {obs['model_size_mb']}mb")

    total_reward = 0.0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step in range(MAX_STEPS):
        if env.done:
            break

        user_prompt = build_prompt(obs, step + 1)
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            response_text = response.choices[0].message.content
            messages.append({
                "role": "assistant",
                "content": response_text
            })
        except Exception as e:
            print(f"  LLM error on step {step + 1}: {e}")
            response_text = '{"type": "noop"}'
            messages.append({
                "role": "assistant",
                "content": response_text
            })

        action = parse_action(response_text)
        obs_obj, reward, done, info = env.step(action)
        obs = obs_obj.model_dump()
        total_reward += reward

        print(
            f"  Step {step+1:2d} | "
            f"Action: {action.type:20s} | "
            f"Reward: {reward:+.3f} | "
            f"Infected: {obs['infected']:3d} | "
            f"Model: {obs['model_size_mb']:6.1f}mb"
        )

        time.sleep(0.5)

    result = env.grade()
    print(f"\nFinal Score: {result.final_score}")
    print(f"Passed: {result.passed}")
    print(f"Breakdown: {result.breakdown}")
    print(f"Feedback: {result.feedback}")

    return {
        "task_id": task_id,
        "final_score": result.final_score,
        "passed": result.passed,
        "total_reward": round(total_reward, 3),
        "breakdown": result.breakdown,
        "feedback": result.feedback
    }


def main():
    print("IndiaShield-v1 Baseline Inference")
    print("Model:", MODEL_NAME)
    print("API:", API_BASE_URL)
    print()

    task_ids = ["task1", "task2", "task3"]
    results = []

    for task_id in task_ids:
        result = run_task(task_id)
        results.append(result)

    print(f"\n{'='*50}")
    print("BASELINE SCORES SUMMARY")
    print(f"{'='*50}")
    total = 0.0
    for r in results:
        print(
            f"{r['task_id']:8s} | "
            f"Score: {r['final_score']:.3f} | "
            f"Passed: {str(r['passed']):5s} | "
            f"{r['feedback'][:40]}"
        )
        total += r["final_score"]
    avg = round(total / len(results), 3)
    print(f"\nAverage score across 3 tasks: {avg}")
    print()

    with open("baseline_scores.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "results": results,
            "average_score": avg
        }, f, indent=2)
    print("Scores saved to baseline_scores.json")


if __name__ == "__main__":
    main()