from typing import Dict
from pydantic import BaseModel
from indiashield.tasks import TaskConfig


class GraderResult(BaseModel):
    task_id: str
    final_score: float
    passed: bool
    breakdown: Dict[str, float]
    feedback: str


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def grade_task1(network_state, model_state, task: TaskConfig) -> GraderResult:
    wc = task.win_condition

    infected = network_state.infected
    total = network_state.total_users
    containment_score = _clamp(
        1.0 - (infected / total)
    )

    if model_state.size_mb <= wc.max_model_size_mb:
        compression_score = 1.0
    else:
        compression_score = _clamp(
            wc.max_model_size_mb / model_state.size_mb
        )

    deployment_score = _clamp(
        model_state.nodes_protected / wc.min_nodes_protected
    )

    source_score = _clamp(
        network_state.sources_found / network_state.total_sources
    )

    final_score = round(
        containment_score * 0.35 +
        compression_score * 0.25 +
        deployment_score * 0.25 +
        source_score * 0.15,
        3
    )

    passed = (
        infected <= wc.max_infected and
        model_state.size_mb <= wc.max_model_size_mb and
        model_state.nodes_protected >= wc.min_nodes_protected and
        network_state.sources_found >= wc.min_sources_found
    )

    feedback_parts = []
    if infected > wc.max_infected:
        feedback_parts.append(
            f"Too many infected ({infected} > {wc.max_infected})"
        )
    if model_state.size_mb > wc.max_model_size_mb:
        feedback_parts.append(
            f"Model too large ({model_state.size_mb}mb > {wc.max_model_size_mb}mb)"
        )
    if model_state.nodes_protected < wc.min_nodes_protected:
        feedback_parts.append(
            f"Not enough nodes protected ({model_state.nodes_protected} < {wc.min_nodes_protected})"
        )
    if not feedback_parts:
        feedback_parts.append("All win conditions met")

    return GraderResult(
        task_id="task1",
        final_score=final_score,
        passed=passed,
        breakdown={
            "containment": round(containment_score, 3),
            "compression": round(compression_score, 3),
            "deployment": round(deployment_score, 3),
            "source_detection": round(source_score, 3),
        },
        feedback=" | ".join(feedback_parts)
    )


def grade_task2(network_state, model_state, task: TaskConfig) -> GraderResult:
    wc = task.win_condition

    infected = network_state.infected
    total = network_state.total_users
    containment_score = _clamp(1.0 - (infected / total))

    if model_state.size_mb <= wc.max_model_size_mb:
        compression_score = 1.0
    else:
        compression_score = _clamp(
            wc.max_model_size_mb / model_state.size_mb
        )

    hindi_score = _clamp(
        model_state.hindi_accuracy / wc.min_hindi_accuracy
    )
    tamil_score = _clamp(
        model_state.tamil_accuracy / wc.min_tamil_accuracy
    )
    accuracy_score = (hindi_score + tamil_score) / 2.0

    deployment_score = _clamp(
        model_state.nodes_protected / wc.min_nodes_protected
    )

    source_score = _clamp(
        network_state.sources_found / network_state.total_sources
    )

    accuracy_penalty = 0.0
    if model_state.hindi_accuracy < wc.min_hindi_accuracy:
        accuracy_penalty += 0.2
    if model_state.tamil_accuracy < wc.min_tamil_accuracy:
        accuracy_penalty += 0.2

    final_score = round(
        _clamp(
            containment_score * 0.30 +
            compression_score * 0.20 +
            accuracy_score * 0.25 +
            deployment_score * 0.15 +
            source_score * 0.10 -
            accuracy_penalty
        ),
        3
    )

    passed = (
        infected <= wc.max_infected and
        model_state.size_mb <= wc.max_model_size_mb and
        model_state.nodes_protected >= wc.min_nodes_protected and
        model_state.hindi_accuracy >= wc.min_hindi_accuracy and
        model_state.tamil_accuracy >= wc.min_tamil_accuracy and
        network_state.sources_found >= wc.min_sources_found
    )

    feedback_parts = []
    if infected > wc.max_infected:
        feedback_parts.append(f"Too many infected ({infected})")
    if model_state.hindi_accuracy < wc.min_hindi_accuracy:
        feedback_parts.append(
            f"Hindi accuracy too low ({model_state.hindi_accuracy})"
        )
    if model_state.tamil_accuracy < wc.min_tamil_accuracy:
        feedback_parts.append(
            f"Tamil accuracy too low ({model_state.tamil_accuracy})"
        )
    if not feedback_parts:
        feedback_parts.append("All win conditions met")

    return GraderResult(
        task_id="task2",
        final_score=final_score,
        passed=passed,
        breakdown={
            "containment": round(containment_score, 3),
            "compression": round(compression_score, 3),
            "accuracy": round(accuracy_score, 3),
            "deployment": round(deployment_score, 3),
            "source_detection": round(source_score, 3),
            "accuracy_penalty": round(accuracy_penalty, 3),
        },
        feedback=" | ".join(feedback_parts)
    )


def grade_task3(network_state, model_state, task: TaskConfig) -> GraderResult:
    wc = task.win_condition

    if network_state.sources_found < network_state.total_sources:
        source_score = _clamp(
            network_state.sources_found / network_state.total_sources
        )
        final_score = round(source_score * 0.3, 3)
        return GraderResult(
            task_id="task3",
            final_score=final_score,
            passed=False,
            breakdown={
                "source_detection": round(source_score, 3),
                "containment": 0.0,
                "compression": 0.0,
                "accuracy": 0.0,
                "deployment": 0.0,
            },
            feedback=(
                f"Not all sources found "
                f"({network_state.sources_found}/{network_state.total_sources}). "
                f"Score capped at 0.3."
            )
        )

    infected = network_state.infected
    total = network_state.total_users
    containment_score = _clamp(1.0 - (infected / total))

    if model_state.size_mb <= wc.max_model_size_mb:
        compression_score = 1.0
    else:
        compression_score = _clamp(
            wc.max_model_size_mb / model_state.size_mb
        )

    hindi_score = _clamp(
        model_state.hindi_accuracy / wc.min_hindi_accuracy
    )
    tamil_score = _clamp(
        model_state.tamil_accuracy / wc.min_tamil_accuracy
    )
    telugu_score = _clamp(
        model_state.telugu_accuracy / 0.75
    )
    accuracy_score = (hindi_score + tamil_score + telugu_score) / 3.0

    deployment_score = _clamp(
        model_state.nodes_protected / wc.min_nodes_protected
    )

    health_outcome_score = _clamp(
        1.0 - (infected / total) * 1.5
    )

    accuracy_penalty = 0.0
    if model_state.hindi_accuracy < wc.min_hindi_accuracy:
        accuracy_penalty += 0.15
    if model_state.tamil_accuracy < wc.min_tamil_accuracy:
        accuracy_penalty += 0.15
    if model_state.telugu_accuracy < 0.75:
        accuracy_penalty += 0.10

    final_score = round(
        _clamp(
            containment_score * 0.25 +
            compression_score * 0.20 +
            accuracy_score * 0.20 +
            deployment_score * 0.20 +
            health_outcome_score * 0.15 -
            accuracy_penalty
        ),
        3
    )

    passed = (
        infected <= wc.max_infected and
        model_state.size_mb <= wc.max_model_size_mb and
        model_state.nodes_protected >= wc.min_nodes_protected and
        model_state.hindi_accuracy >= wc.min_hindi_accuracy and
        model_state.tamil_accuracy >= wc.min_tamil_accuracy and
        network_state.sources_found >= wc.min_sources_found
    )

    feedback_parts = []
    if infected > wc.max_infected:
        feedback_parts.append(f"Too many infected ({infected})")
    if model_state.size_mb > wc.max_model_size_mb:
        feedback_parts.append(f"Model too large ({model_state.size_mb}mb)")
    if model_state.nodes_protected < wc.min_nodes_protected:
        feedback_parts.append(
            f"Not enough nodes protected ({model_state.nodes_protected})"
        )
    if not feedback_parts:
        feedback_parts.append("All win conditions met")

    return GraderResult(
        task_id="task3",
        final_score=final_score,
        passed=passed,
        breakdown={
            "source_detection": 1.0,
            "containment": round(containment_score, 3),
            "compression": round(compression_score, 3),
            "accuracy": round(accuracy_score, 3),
            "deployment": round(deployment_score, 3),
            "health_outcome": round(health_outcome_score, 3),
            "accuracy_penalty": round(accuracy_penalty, 3),
        },
        feedback=" | ".join(feedback_parts)
    )


def grade_task4(network_state, model_state, task: TaskConfig) -> GraderResult:
    wc = task.win_condition

    infected = network_state.infected
    total = network_state.total_users
    containment_score = _clamp(1.0 - (infected / total))

    if model_state.size_mb <= wc.max_model_size_mb:
        compression_score = 1.0
    else:
        compression_score = _clamp(
            wc.max_model_size_mb / model_state.size_mb
        )

    hindi_score = _clamp(
        model_state.hindi_accuracy / wc.min_hindi_accuracy
    )
    tamil_score = _clamp(
        model_state.tamil_accuracy / wc.min_tamil_accuracy
    )
    accuracy_score = (hindi_score + tamil_score) / 2.0

    deployment_score = _clamp(
        model_state.nodes_protected / wc.min_nodes_protected
    )

    source_score = _clamp(
        network_state.sources_found / network_state.total_sources
    )

    speed_bonus = 0.0
    turns_used = network_state.time_elapsed
    if turns_used <= task.max_turns * 0.6:
        speed_bonus = 0.1

    final_score = round(
        _clamp(
            containment_score * 0.28 +
            compression_score * 0.20 +
            accuracy_score * 0.20 +
            deployment_score * 0.17 +
            source_score * 0.15 +
            speed_bonus
        ),
        3
    )

    passed = (
        infected <= wc.max_infected and
        model_state.size_mb <= wc.max_model_size_mb and
        model_state.nodes_protected >= wc.min_nodes_protected and
        network_state.sources_found >= wc.min_sources_found
    )

    feedback_parts = []
    if infected > wc.max_infected:
        feedback_parts.append(f"Too many infected ({infected})")
    if model_state.size_mb > wc.max_model_size_mb:
        feedback_parts.append(f"Model too large ({model_state.size_mb}mb)")
    if not feedback_parts:
        feedback_parts.append("All win conditions met")

    return GraderResult(
        task_id="task4",
        final_score=final_score,
        passed=passed,
        breakdown={
            "containment": round(containment_score, 3),
            "compression": round(compression_score, 3),
            "accuracy": round(accuracy_score, 3),
            "deployment": round(deployment_score, 3),
            "source_detection": round(source_score, 3),
            "speed_bonus": round(speed_bonus, 3),
        },
        feedback=" | ".join(feedback_parts)
    )


def grade_task5(network_state, model_state, task: TaskConfig) -> GraderResult:
    wc = task.win_condition

    if network_state.sources_found < network_state.total_sources:
        source_score = _clamp(
            network_state.sources_found / network_state.total_sources
        )
        final_score = round(source_score * 0.25, 3)
        return GraderResult(
            task_id="task5",
            final_score=final_score,
            passed=False,
            breakdown={
                "source_detection": round(source_score, 3),
                "containment": 0.0,
                "compression": 0.0,
                "accuracy": 0.0,
                "deployment": 0.0,
            },
            feedback=(
                f"Not all sources found "
                f"({network_state.sources_found}/{network_state.total_sources}). "
                f"Score capped. In this scenario every missed source "
                f"represents potential real world harm."
            )
        )

    infected = network_state.infected
    total = network_state.total_users
    containment_score = _clamp(1.0 - (infected / total))

    if model_state.size_mb <= wc.max_model_size_mb:
        compression_score = 1.0
    else:
        compression_score = _clamp(
            wc.max_model_size_mb / model_state.size_mb
        )

    hindi_score = _clamp(
        model_state.hindi_accuracy / wc.min_hindi_accuracy
    )
    tamil_score = _clamp(
        model_state.tamil_accuracy / wc.min_tamil_accuracy
    )
    telugu_score = _clamp(
        model_state.telugu_accuracy / 0.78
    )
    accuracy_score = (hindi_score + tamil_score + telugu_score) / 3.0

    deployment_score = _clamp(
        model_state.nodes_protected / wc.min_nodes_protected
    )

    false_positive_penalty = 0.0
    blocked = network_state.blocked
    actual_misinfo = network_state.infected + network_state.blocked
    if actual_misinfo > 0:
        false_positive_rate = max(
            0.0,
            (blocked - network_state.sources_found) / max(1, actual_misinfo)
        )
        false_positive_penalty = false_positive_rate * 0.3

    final_score = round(
        _clamp(
            containment_score * 0.25 +
            compression_score * 0.15 +
            accuracy_score * 0.25 +
            deployment_score * 0.20 +
            0.15 -
            false_positive_penalty
        ),
        3
    )

    passed = (
        infected <= wc.max_infected and
        model_state.size_mb <= wc.max_model_size_mb and
        model_state.nodes_protected >= wc.min_nodes_protected and
        model_state.hindi_accuracy >= wc.min_hindi_accuracy and
        model_state.tamil_accuracy >= wc.min_tamil_accuracy and
        network_state.sources_found >= wc.min_sources_found
    )

    feedback_parts = []
    if infected > wc.max_infected:
        feedback_parts.append(f"Too many infected ({infected})")
    if false_positive_penalty > 0.1:
        feedback_parts.append(
            f"High false positive rate — legitimate speech suppressed"
        )
    if model_state.nodes_protected < wc.min_nodes_protected:
        feedback_parts.append(
            f"Not enough nodes protected ({model_state.nodes_protected})"
        )
    if not feedback_parts:
        feedback_parts.append("All win conditions met")

    return GraderResult(
        task_id="task5",
        final_score=final_score,
        passed=passed,
        breakdown={
            "source_detection": 1.0,
            "containment": round(containment_score, 3),
            "compression": round(compression_score, 3),
            "accuracy": round(accuracy_score, 3),
            "deployment": round(deployment_score, 3),
            "false_positive_penalty": round(false_positive_penalty, 3),
        },
        feedback=" | ".join(feedback_parts)
    )


GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
    "task4": grade_task4,
    "task5": grade_task5,
}


def grade(
    task_id: str,
    network_state,
    model_state,
    task: TaskConfig
) -> GraderResult:
    grader_fn = GRADERS.get(task_id)
    if grader_fn is None:
        return GraderResult(
            task_id=task_id,
            final_score=0.0,
            passed=False,
            breakdown={},
            feedback=f"Unknown task id: {task_id}"
        )
    return grader_fn(network_state, model_state, task)