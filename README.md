

# IndiaShield-v1

A reinforcement learning environment where an agent contains WhatsApp misinformation outbreaks in India while compressing a multilingual MuRIL classifier for low-RAM Indian devices.

## Tasks

| Task | Story | Difficulty |
|---|---|---|
| task1 | The Diwali Cracker Rumour | Easy |
| task2 | The Election EVM Story | Medium |
| task3 | The Hospital Poison Campaign | Hard |
| task4 | The IPL Match Fixing Rumour | Very Hard |
| task5 | The Religious Violence Incitement | Expert |

## API

- POST /reset — start a new episode
- POST /step — take one action
- GET /state — current state
- GET /grade — current score
- GET /docs — interactive API documentation

## Actions

- intercept — block one infected node
- quarantine — block an entire WhatsApp group
- identify_spreader — find the highest-risk node
- add_forward_label — mark message as forwarded many times
- quantize — shrink model by reducing math precision
- prune — cut unused model weights
- distill — rebuild into tiny student model
- deploy — install compressed model on a phone

## Baseline Scores

| Task | Score | Passed |
|---|---|---|
| task1 | 0.688 | ✓ |
| task2 | 0.335 | ✗ |
| task3 | 0.000 | ✗ |

Average: 0.341 (Qwen2.5-7B-Instruct baseline)

Note: task1 passed on first try in 3 steps. task3 scores 0.0 by design — expert difficulty requires finding all 3 coordinated sources which genuinely challenges frontier models.

## Built for

OpenEnv Hackathon — Meta x HuggingFace
