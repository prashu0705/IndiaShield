from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel
from indiashield.network import WhatsAppNetwork
from indiashield.model import MuRILModel
from indiashield.tasks import get_task, TaskConfig
from indiashield.graders import grade, GraderResult
import time
import random


class Action(BaseModel):
    type: str
    node_id: Optional[int] = None
    group_id: Optional[int] = None
    precision: Optional[str] = "int8"
    target_layer: Optional[str] = "ffn"
    percentage: Optional[int] = 30
    student_size: Optional[str] = "small"


class Observation(BaseModel):
    turn: int
    max_turns: int
    total_users: int
    infected: int
    clean: int
    blocked: int
    sources_found: int
    total_sources: int
    super_spreaders_found: int
    total_super_spreaders: int
    model_size_mb: float
    target_size_mb: float
    hindi_accuracy: float
    english_accuracy: float
    tamil_accuracy: float
    telugu_accuracy: float
    overall_accuracy: float
    nodes_protected: int
    distill_turns_remaining: int
    model_ready_to_deploy: bool
    last_action_result: Dict[str, Any] = {}
    task_id: str
    task_name: str
    task_story: str
    episode_id: str
    created_at: float


class EpisodeStats(BaseModel):
    episode_id: str
    task_id: str
    total_turns: int
    total_reward: float
    final_score: float
    passed: bool
    duration_seconds: float


from openenv.env import Env

class IndiaShieldEnv(Env):

    def __init__(
        self,
        task_id: str = "task1",
        seed: Optional[int] = None,
        custom_config: Optional[Dict] = None
    ):
        self.task_id = task_id
        self.seed = seed if seed is not None else random.randint(0, 99999)
        self.custom_config = custom_config or {}
        self.task: Optional[TaskConfig] = None
        self.network: Optional[WhatsAppNetwork] = None
        self.model: Optional[MuRILModel] = None
        self.turn: int = 0
        self.done: bool = False
        self.last_action_result: Dict[str, Any] = {}
        self.last_reward: float = 0.0
        self.total_reward: float = 0.0
        self.episode_id: str = self._make_episode_id()
        self.created_at: float = time.time()
        self.name = "IndiaShield-v1"
        self.state_space = {
            "infected": "int",
            "clean": "int", 
            "blocked": "int",
            "model_size_mb": "float",
            "hindi_accuracy": "float",
            "english_accuracy": "float",
            "tamil_accuracy": "float",
            "telugu_accuracy": "float",
            "nodes_protected": "int",
            "sources_found": "int",
            "turn": "int"
        }
        self.action_space = {
            "type": ["intercept", "quarantine", "identify_spreader",
                     "add_forward_label", "quantize", "prune",
                     "distill", "deploy", "noop"],
            "node_id": "int (optional)",
            "group_id": "int (optional)",
            "precision": ["int8", "int4"],
            "target_layer": ["attention", "ffn", "all"],
            "percentage": "int 0-70",
            "student_size": ["small", "tiny"]
        }
        self.episode_max_length = self.custom_config.get(
            "max_turns", 20
        )
        self.reset()

    def _make_episode_id(self) -> str:
        return f"{self.task_id}_{self.seed}_{int(time.time())}"

    def reset(self) -> Observation:
        self.task = get_task(self.task_id)

        total_users = self.custom_config.get(
            "total_users", self.task.total_users
        )
        num_groups = self.custom_config.get(
            "num_groups", self.task.num_groups
        )
        max_turns = self.custom_config.get(
            "max_turns", self.task.max_turns
        )
        initial_model_size = self.custom_config.get(
            "initial_model_size_mb", self.task.initial_model_size_mb
        )
        target_model_size = self.custom_config.get(
            "target_model_size_mb", self.task.target_model_size_mb
        )

        self.task.max_turns = max_turns

        self.network = WhatsAppNetwork(
            total_users=total_users,
            num_groups=num_groups,
            num_super_spreaders=self.task.num_super_spreaders,
            num_sources=self.task.num_sources,
            task_id=self.task.task_id,
            max_time=max_turns,
            seed=self.seed
        )
        self.model = MuRILModel(
            initial_size_mb=initial_model_size,
            target_size_mb=target_model_size,
            language_mix=self.task.language_mix,
            seed=self.seed
        )
        self.turn = 0
        self.done = False
        self.last_action_result = {}
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.episode_id = self._make_episode_id()
        self.created_at = time.time()
        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        if self.done:
            obs = self._get_observation()
            return obs, 0.0, True, {
                "message": "Episode already done. Call reset().",
                "episode_id": self.episode_id
            }

        prev_network_state = self.network.get_state()
        prev_model_state = self.model.get_state()

        action_result = self._execute_action(action)
        self.last_action_result = action_result

        if self.model.distill_turns_remaining == 0:
            self.network.spread()

        self.turn += 1
        reward = self._calculate_reward(
            prev_network_state,
            prev_model_state,
            action_result
        )
        self.last_reward = reward
        self.total_reward = round(self.total_reward + reward, 3)
        self.done = self._is_done()

        obs = self._get_observation()
        info = {
            "action_result": action_result,
            "reward": reward,
            "total_reward": self.total_reward,
            "turn": self.turn,
            "done": self.done,
            "episode_id": self.episode_id
        }
        return obs, reward, self.done, info

    def state(self) -> Dict:
        net_state = self.network.get_state()
        mod_state = self.model.get_state()
        return {
            "episode_id": self.episode_id,
            "turn": self.turn,
            "max_turns": self.task.max_turns,
            "network": net_state.model_dump(),
            "model": mod_state.model_dump(),
            "done": self.done,
            "total_reward": self.total_reward,
            "task_id": self.task_id,
            "seed": self.seed,
            "created_at": self.created_at,
        }

    def grade(self) -> GraderResult:
        net_state = self.network.get_state()
        mod_state = self.model.get_state()
        return grade(self.task_id, net_state, mod_state, self.task)

    def get_stats(self) -> EpisodeStats:
        result = self.grade()
        return EpisodeStats(
            episode_id=self.episode_id,
            task_id=self.task_id,
            total_turns=self.turn,
            total_reward=self.total_reward,
            final_score=result.final_score,
            passed=result.passed,
            duration_seconds=round(time.time() - self.created_at, 2)
        )

    def _execute_action(self, action: Action) -> Dict:
        if action.type == "intercept":
            if action.node_id is None:
                return {"success": False, "reason": "node_id required"}
            success = self.network.intercept(action.node_id)
            return {
                "success": success,
                "action": "intercept",
                "node_id": action.node_id
            }

        elif action.type == "quarantine":
            if action.group_id is None:
                return {"success": False, "reason": "group_id required"}
            blocked = self.network.quarantine(action.group_id)
            return {
                "success": True,
                "action": "quarantine",
                "group_id": action.group_id,
                "nodes_blocked": blocked
            }

        elif action.type == "identify_spreader":
            result = self.network.identify_spreader()
            if result is None:
                return {
                    "success": False,
                    "reason": "No infected nodes found"
                }
            node_id, message_id = result
            node = self.network.nodes[node_id]
            return {
                "success": True,
                "action": "identify_spreader",
                "node_id": node_id,
                "message_id": message_id,
                "is_super_spreader": node.is_super_spreader,
                "is_source": node.is_source,
                "connections": len(node.connections)
            }

        elif action.type == "add_forward_label":
            if action.node_id is None:
                return {"success": False, "reason": "node_id required"}
            success = self.network.add_forward_label(action.node_id)
            return {
                "success": success,
                "action": "add_forward_label",
                "node_id": action.node_id
            }

        elif action.type == "quantize":
            precision = action.precision or "int8"
            return self.model.quantize(precision)

        elif action.type == "prune":
            layer = action.target_layer or "ffn"
            pct = action.percentage or 30
            return self.model.prune(layer, pct)

        elif action.type == "distill":
            size = action.student_size or "small"
            return self.model.distill(size)

        elif action.type == "deploy":
            if action.node_id is None:
                return {"success": False, "reason": "node_id required"}
            return self.model.deploy(action.node_id)

        elif action.type == "noop":
            return {"success": True, "action": "noop"}

        else:
            return {
                "success": False,
                "reason": f"Unknown action type: {action.type}"
            }

    def _calculate_reward(
        self,
        prev_network_state,
        prev_model_state,
        action_result: Dict
    ) -> float:
        r = 0.0
        net = self.network.get_state()
        mod = self.model.get_state()

        time_weight = 1.0 - (self.turn / self.task.max_turns) * 0.3

        new_infections = net.infected - prev_network_state.infected
        if new_infections < 0:
            r += 0.4 * time_weight
        elif new_infections == 0:
            r += 0.1 * time_weight
        else:
            r -= new_infections * 0.05

        if mod.size_mb < prev_model_state.size_mb:
            compression_ratio = (
                prev_model_state.size_mb - mod.size_mb
            ) / prev_model_state.size_mb
            r += compression_ratio * 0.3

        hindi_drop = prev_model_state.hindi_accuracy - mod.hindi_accuracy
        english_drop = (
            prev_model_state.english_accuracy - mod.english_accuracy
        )
        r -= hindi_drop * 0.4
        r -= english_drop * 0.2

        new_protected = (
            mod.nodes_protected - prev_model_state.nodes_protected
        )
        r += new_protected * 0.02

        if (
            action_result.get("action") == "identify_spreader" and
            action_result.get("is_super_spreader")
        ):
            r += 0.3

        if action_result.get("action") == "add_forward_label":
            r += 0.1

        if action_result.get("action") == "noop":
            r -= 0.15

        if not action_result.get("success", True):
            r -= 0.1

        return round(max(-1.0, min(1.0, r)), 3)

    def _is_done(self) -> bool:
        if self.turn >= self.task.max_turns:
            return True
        net = self.network.get_state()
        if net.infected == 0:
            return True
        return False

    def _get_observation(self) -> Observation:
        net = self.network.get_state()
        mod = self.model.get_state()
        return Observation(
            turn=self.turn,
            max_turns=self.task.max_turns,
            total_users=net.total_users,
            infected=net.infected,
            clean=net.clean,
            blocked=net.blocked,
            sources_found=net.sources_found,
            total_sources=net.total_sources,
            super_spreaders_found=net.super_spreaders_found,
            total_super_spreaders=net.total_super_spreaders,
            model_size_mb=mod.size_mb,
            target_size_mb=mod.target_size_mb,
            hindi_accuracy=mod.hindi_accuracy,
            english_accuracy=mod.english_accuracy,
            tamil_accuracy=mod.tamil_accuracy,
            telugu_accuracy=mod.telugu_accuracy,
            overall_accuracy=mod.overall_accuracy,
            nodes_protected=mod.nodes_protected,
            distill_turns_remaining=mod.distill_turns_remaining,
            model_ready_to_deploy=self.model.is_ready_to_deploy(),
            last_action_result=self.last_action_result,
            task_id=self.task_id,
            task_name=self.task.name,
            task_story=self.task.story,
            episode_id=self.episode_id,
            created_at=self.created_at
        )