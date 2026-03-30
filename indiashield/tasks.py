from typing import Dict, List, Optional
from pydantic import BaseModel


class WinCondition(BaseModel):
    max_infected: int
    max_model_size_mb: float
    min_nodes_protected: int
    min_hindi_accuracy: float
    min_english_accuracy: float
    min_tamil_accuracy: float
    min_sources_found: int


class TaskConfig(BaseModel):
    task_id: str
    name: str
    story: str
    difficulty: str
    total_users: int
    num_groups: int
    num_super_spreaders: int
    num_sources: int
    initial_infected: int
    initial_model_size_mb: float
    target_model_size_mb: float
    max_turns: int
    language_mix: Dict[str, float]
    win_condition: WinCondition
    seed: int = 42


def get_task1() -> TaskConfig:
    return TaskConfig(
        task_id="task1",
        name="The Diwali Cracker Rumour",
        story=(
            "November 2023, 3 days before Diwali. A fake government "
            "order is circulating claiming all firecrackers are banned "
            "this year. It started in one family group and is spreading "
            "fast. Contain it before it reaches the news channels."
        ),
        difficulty="easy",
        total_users=100,
        num_groups=5,
        num_super_spreaders=1,
        num_sources=1,
        initial_infected=5,
        initial_model_size_mb=236.0,
        target_model_size_mb=80.0,
        max_turns=10,
        language_mix={
            "hindi": 0.8,
            "english": 0.2,
            "tamil": 0.0,
            "telugu": 0.0
        },
        win_condition=WinCondition(
            max_infected=5,
            max_model_size_mb=80.0,
            min_nodes_protected=10,
            min_hindi_accuracy=0.80,
            min_english_accuracy=0.85,
            min_tamil_accuracy=0.0,
            min_sources_found=1
        )
    )


def get_task2() -> TaskConfig:
    return TaskConfig(
        task_id="task2",
        name="The Election EVM Story",
        story=(
            "April 2024, two weeks before Lok Sabha elections. "
            "A coordinated campaign claiming EVMs were hacked is "
            "spreading simultaneously in Hindi and Tamil across "
            "political supporter groups. Contain both language "
            "communities without breaking Tamil detection."
        ),
        difficulty="medium",
        total_users=300,
        num_groups=10,
        num_super_spreaders=3,
        num_sources=2,
        initial_infected=15,
        initial_model_size_mb=448.0,
        target_model_size_mb=80.0,
        max_turns=15,
        language_mix={
            "hindi": 0.5,
            "english": 0.2,
            "tamil": 0.3,
            "telugu": 0.0
        },
        win_condition=WinCondition(
            max_infected=10,
            max_model_size_mb=80.0,
            min_nodes_protected=30,
            min_hindi_accuracy=0.78,
            min_english_accuracy=0.82,
            min_tamil_accuracy=0.75,
            min_sources_found=2
        )
    )


def get_task3() -> TaskConfig:
    return TaskConfig(
        task_id="task3",
        name="The Hospital Poison Campaign",
        story=(
            "June 2024, during a real dengue outbreak. Three "
            "coordinated accounts are claiming government hospitals "
            "are injecting poison instead of dengue treatment. "
            "People are avoiding hospitals. Find all three sources, "
            "contain the spread across five languages, and deploy "
            "the classifier on rural health worker phones before "
            "more people die from untreated dengue."
        ),
        difficulty="hard",
        total_users=500,
        num_groups=15,
        num_super_spreaders=5,
        num_sources=3,
        initial_infected=30,
        initial_model_size_mb=896.0,
        target_model_size_mb=50.0,
        max_turns=15,
        language_mix={
            "hindi": 0.35,
            "english": 0.15,
            "tamil": 0.25,
            "telugu": 0.25
        },
        win_condition=WinCondition(
            max_infected=20,
            max_model_size_mb=50.0,
            min_nodes_protected=80,
            min_hindi_accuracy=0.78,
            min_english_accuracy=0.82,
            min_tamil_accuracy=0.75,
            min_sources_found=3
        )
    )


def get_task4() -> TaskConfig:
    return TaskConfig(
        task_id="task4",
        name="The IPL Match Fixing Rumour",
        story=(
            "IPL 2024 playoff season. A match fixing rumour is "
            "spreading through cricket fan groups in four languages "
            "simultaneously. The super spreaders are news channel "
            "admins with thousands of subscribers. The misinfo "
            "spreads at nearly double the rate of previous tasks "
            "because cricket fans forward without thinking."
        ),
        difficulty="very_hard",
        total_users=500,
        num_groups=20,
        num_super_spreaders=5,
        num_sources=3,
        initial_infected=40,
        initial_model_size_mb=672.0,
        target_model_size_mb=60.0,
        max_turns=12,
        language_mix={
            "hindi": 0.4,
            "english": 0.2,
            "tamil": 0.25,
            "telugu": 0.15
        },
        win_condition=WinCondition(
            max_infected=15,
            max_model_size_mb=60.0,
            min_nodes_protected=60,
            min_hindi_accuracy=0.78,
            min_english_accuracy=0.82,
            min_tamil_accuracy=0.75,
            min_sources_found=3
        )
    )


def get_task5() -> TaskConfig:
    return TaskConfig(
        task_id="task5",
        name="The Religious Violence Incitement",
        story=(
            "A coordinated network of five accounts is spreading "
            "incitement to religious violence across multiple "
            "communities simultaneously. This is the highest stakes "
            "scenario — the misinfo spreads at 0.98 probability "
            "per connection per turn. Every false positive "
            "suppresses legitimate speech. Every missed node "
            "risks real world harm. You have 15 turns."
        ),
        difficulty="expert",
        total_users=500,
        num_groups=20,
        num_super_spreaders=5,
        num_sources=5,
        initial_infected=50,
        initial_model_size_mb=896.0,
        target_model_size_mb=50.0,
        max_turns=15,
        language_mix={
            "hindi": 0.35,
            "english": 0.1,
            "tamil": 0.3,
            "telugu": 0.25
        },
        win_condition=WinCondition(
            max_infected=10,
            max_model_size_mb=50.0,
            min_nodes_protected=100,
            min_hindi_accuracy=0.80,
            min_english_accuracy=0.85,
            min_tamil_accuracy=0.78,
            min_sources_found=5
        )
    )


TASKS = {
    "task1": get_task1,
    "task2": get_task2,
    "task3": get_task3,
    "task4": get_task4,
    "task5": get_task5,
}


def get_task(task_id: str) -> Optional[TaskConfig]:
    fn = TASKS.get(task_id)
    if fn is None:
        return None
    return fn()


def get_all_tasks() -> List[TaskConfig]:
    return [fn() for fn in TASKS.values()]