import random
from typing import Dict, List, Optional
from pydantic import BaseModel


class ModelState(BaseModel):
    size_mb: float
    target_size_mb: float
    hindi_accuracy: float
    english_accuracy: float
    tamil_accuracy: float
    telugu_accuracy: float
    overall_accuracy: float
    nodes_protected: int
    distill_turns_remaining: int
    compression_history: List[str] = []


class MuRILModel:

    def __init__(
        self,
        initial_size_mb: float,
        target_size_mb: float,
        language_mix: Dict[str, float],
        seed: int = 42
    ):
        self.size_mb = initial_size_mb
        self.initial_size_mb = initial_size_mb
        self.target_size_mb = target_size_mb
        self.language_mix = language_mix
        self.rng = random.Random(seed)

        self.hindi_accuracy = 0.91
        self.english_accuracy = 0.93
        self.tamil_accuracy = 0.88
        self.telugu_accuracy = 0.86

        self.nodes_protected = 0
        self.distill_turns_remaining = 0
        self.compression_history: List[str] = []

        self._apply_language_mix_penalty()

    def _apply_language_mix_penalty(self):
        hindi_weight = self.language_mix.get("hindi", 0.0)
        tamil_weight = self.language_mix.get("tamil", 0.0)
        telugu_weight = self.language_mix.get("telugu", 0.0)

        if hindi_weight > 0.5:
            self.hindi_accuracy += 0.02
        if tamil_weight > 0.2:
            self.tamil_accuracy += 0.01
        if telugu_weight > 0.1:
            self.telugu_accuracy += 0.01

    def _update_overall_accuracy(self):
        weights = {
            "hindi": self.language_mix.get("hindi", 0.25),
            "english": self.language_mix.get("english", 0.25),
            "tamil": self.language_mix.get("tamil", 0.25),
            "telugu": self.language_mix.get("telugu", 0.25),
        }
        total_weight = sum(weights.values())
        self.overall_accuracy = (
            self.hindi_accuracy * weights["hindi"] +
            self.english_accuracy * weights["english"] +
            self.tamil_accuracy * weights["tamil"] +
            self.telugu_accuracy * weights["telugu"]
        ) / total_weight

    def quantize(self, precision: str = "int8") -> Dict:
        if precision == "int8":
            size_reduction = 0.55
            hindi_drop = 0.03
            english_drop = 0.02
            tamil_drop = 0.04
            telugu_drop = 0.04
            label = "quantize_int8"

        elif precision == "int4":
            size_reduction = 0.72
            hindi_drop = 0.12
            english_drop = 0.06
            tamil_drop = 0.16
            telugu_drop = 0.18
            label = "quantize_int4"

        else:
            return {"success": False, "reason": "unknown precision"}

        old_size = self.size_mb
        self.size_mb = round(self.size_mb * (1.0 - size_reduction), 1)
        self.hindi_accuracy = round(
            max(0.0, self.hindi_accuracy - hindi_drop), 3
        )
        self.english_accuracy = round(
            max(0.0, self.english_accuracy - english_drop), 3
        )
        self.tamil_accuracy = round(
            max(0.0, self.tamil_accuracy - tamil_drop), 3
        )
        self.telugu_accuracy = round(
            max(0.0, self.telugu_accuracy - telugu_drop), 3
        )
        self._update_overall_accuracy()
        self.compression_history.append(label)

        return {
            "success": True,
            "action": label,
            "size_before": old_size,
            "size_after": self.size_mb,
            "size_reduction_pct": round(size_reduction * 100, 1),
            "hindi_accuracy": self.hindi_accuracy,
            "english_accuracy": self.english_accuracy,
            "tamil_accuracy": self.tamil_accuracy,
            "telugu_accuracy": self.telugu_accuracy,
        }

    def prune(self, target_layer: str = "ffn", percentage: int = 30) -> Dict:
        percentage = max(0, min(percentage, 70))

        if target_layer == "attention":
            size_reduction = percentage * 0.006
            hindi_drop = percentage * 0.0015
            english_drop = percentage * 0.0007
            tamil_drop = percentage * 0.002
            telugu_drop = percentage * 0.0022
            label = f"prune_attention_{percentage}"

        elif target_layer == "ffn":
            size_reduction = percentage * 0.007
            hindi_drop = percentage * 0.0010
            english_drop = percentage * 0.0008
            tamil_drop = percentage * 0.0012
            telugu_drop = percentage * 0.0013
            label = f"prune_ffn_{percentage}"

        elif target_layer == "all":
            size_reduction = percentage * 0.0085
            hindi_drop = percentage * 0.0020
            english_drop = percentage * 0.0010
            tamil_drop = percentage * 0.0028
            telugu_drop = percentage * 0.0030
            label = f"prune_all_{percentage}"

        else:
            return {"success": False, "reason": "unknown layer"}

        old_size = self.size_mb
        self.size_mb = round(
            self.size_mb * (1.0 - size_reduction), 1
        )
        self.hindi_accuracy = round(
            max(0.0, self.hindi_accuracy - hindi_drop), 3
        )
        self.english_accuracy = round(
            max(0.0, self.english_accuracy - english_drop), 3
        )
        self.tamil_accuracy = round(
            max(0.0, self.tamil_accuracy - tamil_drop), 3
        )
        self.telugu_accuracy = round(
            max(0.0, self.telugu_accuracy - telugu_drop), 3
        )
        self._update_overall_accuracy()
        self.compression_history.append(label)

        return {
            "success": True,
            "action": label,
            "size_before": old_size,
            "size_after": self.size_mb,
            "hindi_accuracy": self.hindi_accuracy,
            "english_accuracy": self.english_accuracy,
            "tamil_accuracy": self.tamil_accuracy,
            "telugu_accuracy": self.telugu_accuracy,
        }

    def distill(self, student_size: str = "small") -> Dict:
        if self.distill_turns_remaining > 0:
            self.distill_turns_remaining -= 1
            if self.distill_turns_remaining == 0:
                return self._complete_distillation()
            return {
                "success": True,
                "action": "distill_in_progress",
                "turns_remaining": self.distill_turns_remaining,
                "message": "Distillation in progress. Cannot take other model actions."
            }

        if student_size == "small":
            self._pending_size = round(self.size_mb * 0.35, 1)
            self._pending_hindi_drop = 0.05
            self._pending_english_drop = 0.04
            self._pending_tamil_drop = 0.07
            self._pending_telugu_drop = 0.08
            self._pending_label = "distill_small"
            turns_needed = 2

        elif student_size == "tiny":
            self._pending_size = round(self.size_mb * 0.18, 1)
            self._pending_hindi_drop = 0.09
            self._pending_english_drop = 0.07
            self._pending_tamil_drop = 0.14
            self._pending_telugu_drop = 0.16
            self._pending_label = "distill_tiny"
            turns_needed = 3

        else:
            return {"success": False, "reason": "unknown student size"}

        self.distill_turns_remaining = turns_needed - 1
        return {
            "success": True,
            "action": "distill_started",
            "student_size": student_size,
            "turns_remaining": self.distill_turns_remaining,
            "message": f"Distillation started. Will complete in {turns_needed} turns."
        }

    def _complete_distillation(self) -> Dict:
        old_size = self.size_mb
        self.size_mb = self._pending_size
        self.hindi_accuracy = round(
            max(0.0, self.hindi_accuracy - self._pending_hindi_drop), 3
        )
        self.english_accuracy = round(
            max(0.0, self.english_accuracy - self._pending_english_drop), 3
        )
        self.tamil_accuracy = round(
            max(0.0, self.tamil_accuracy - self._pending_tamil_drop), 3
        )
        self.telugu_accuracy = round(
            max(0.0, self.telugu_accuracy - self._pending_telugu_drop), 3
        )
        self._update_overall_accuracy()
        self.compression_history.append(self._pending_label)

        return {
            "success": True,
            "action": self._pending_label,
            "size_before": old_size,
            "size_after": self.size_mb,
            "hindi_accuracy": self.hindi_accuracy,
            "english_accuracy": self.english_accuracy,
            "tamil_accuracy": self.tamil_accuracy,
            "telugu_accuracy": self.telugu_accuracy,
            "message": "Distillation complete."
        }

    def deploy(self, node_id: int) -> Dict:
        if self.size_mb > self.target_size_mb:
            return {
                "success": False,
                "reason": f"Model too large. Size {self.size_mb}mb exceeds target {self.target_size_mb}mb.",
                "size_mb": self.size_mb,
                "target_size_mb": self.target_size_mb
            }
        self.nodes_protected += 1
        return {
            "success": True,
            "action": "deploy",
            "node_id": node_id,
            "nodes_protected": self.nodes_protected,
            "model_size_mb": self.size_mb,
            "overall_accuracy": self.overall_accuracy
        }

    def is_ready_to_deploy(self) -> bool:
        return self.size_mb <= self.target_size_mb

    def get_state(self) -> ModelState:
        self._update_overall_accuracy()
        return ModelState(
            size_mb=round(self.size_mb, 1),
            target_size_mb=self.target_size_mb,
            hindi_accuracy=self.hindi_accuracy,
            english_accuracy=self.english_accuracy,
            tamil_accuracy=self.tamil_accuracy,
            telugu_accuracy=self.telugu_accuracy,
            overall_accuracy=round(self.overall_accuracy, 3),
            nodes_protected=self.nodes_protected,
            distill_turns_remaining=self.distill_turns_remaining,
            compression_history=self.compression_history.copy()
        )