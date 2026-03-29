import random
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from indiashield.messages import get_misinfo_for_task, get_random_real_news, get_group_config


class Node(BaseModel):
    id: int
    name: str
    group_id: int
    group_type: str
    has_misinfo: bool = False
    status: str = "clean"
    message_id: Optional[str] = None
    is_super_spreader: bool = False
    is_source: bool = False
    forward_label_applied: bool = False
    connections: List[int] = []

    class Config:
        arbitrary_types_allowed = True


class Group(BaseModel):
    id: int
    name: str
    group_type: str
    member_ids: List[int] = []
    trust_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    skepticism: float = 0.3
    is_quarantined: bool = False


class NetworkState(BaseModel):
    total_users: int
    infected: int
    clean: int
    blocked: int
    super_spreaders_found: int
    total_super_spreaders: int
    sources_found: int
    total_sources: int
    time_elapsed: int
    max_time: int
    forward_labels_applied: int


class WhatsAppNetwork:

    def __init__(
        self,
        total_users: int,
        num_groups: int,
        num_super_spreaders: int,
        num_sources: int,
        task_id: str,
        max_time: int,
        seed: int = 42
    ):
        self.total_users = total_users
        self.num_groups = num_groups
        self.num_super_spreaders = num_super_spreaders
        self.num_sources = num_sources
        self.task_id = task_id
        self.max_time = max_time
        self.seed = seed
        self.rng = random.Random(seed)

        self.nodes: Dict[int, Node] = {}
        self.groups: Dict[int, Group] = {}
        self.time_elapsed: int = 0
        self.super_spreaders_found: int = 0
        self.sources_found: int = 0

        self._build_network()

    def _build_network(self):
        group_types = ["family", "locality", "college", "religious", "news_channel"]

        for g in range(self.num_groups):
            gtype = self.rng.choice(group_types)
            config = get_group_config(gtype)
            group = Group(
                id=g,
                name=f"{config['name_template']} {g+1}",
                group_type=gtype,
                trust_multiplier=config["trust_multiplier"],
                spread_multiplier=config["spread_multiplier"],
                skepticism=config["skepticism"]
            )
            self.groups[g] = group

        indian_names = [
            "Raj", "Priya", "Amit", "Sunita", "Vikram", "Deepa",
            "Rahul", "Anjali", "Suresh", "Kavita", "Arun", "Meena",
            "Vijay", "Lakshmi", "Ravi", "Geeta", "Manoj", "Pooja",
            "Sanjay", "Rekha", "Arjun", "Divya", "Nikhil", "Sneha",
            "Karan", "Neha", "Rohit", "Shweta", "Tarun", "Asha",
            "Venkat", "Padma", "Ganesh", "Uma", "Krishnan", "Saranya",
            "Balaji", "Meenakshi", "Senthil", "Gayathri", "Murugan",
            "Anand", "Bharati", "Selvam", "Kamala", "Dinesh", "Radha",
            "Sunil", "Parvathi", "Ramesh"
        ]

        for i in range(self.total_users):
            group_id = i % self.num_groups
            name = self.rng.choice(indian_names) + f"_{i}"
            node = Node(
                id=i,
                name=name,
                group_id=group_id,
                group_type=self.groups[group_id].group_type
            )
            self.nodes[i] = node
            self.groups[group_id].member_ids.append(i)

        for node_id, node in self.nodes.items():
            same_group = [
                n for n in self.groups[node.group_id].member_ids
                if n != node_id
            ]
            num_connections = min(
                self.rng.randint(3, 8),
                len(same_group)
            )
            node.connections = self.rng.sample(same_group, num_connections)

        all_ids = list(self.nodes.keys())
        spreader_ids = self.rng.sample(all_ids, self.num_super_spreaders)
        for sid in spreader_ids:
            self.nodes[sid].is_super_spreader = True
            extra = self.rng.sample(
                [n for n in all_ids if n != sid],
                min(15, self.total_users - 1)
            )
            self.nodes[sid].connections = list(
                set(self.nodes[sid].connections + extra)
            )

        source_ids = self.rng.sample(all_ids, self.num_sources)
        misinfo_messages = get_misinfo_for_task(self.task_id)

        for i, sid in enumerate(source_ids):
            msg = misinfo_messages[i % len(misinfo_messages)]
            self.nodes[sid].is_source = True
            self.nodes[sid].has_misinfo = True
            self.nodes[sid].status = "infected"
            self.nodes[sid].message_id = msg["id"]

    def spread(self):
        newly_infected = []

        for node_id, node in self.nodes.items():
            if node.status != "infected":
                continue

            group = self.groups[node.group_id]
            if group.is_quarantined:
                continue

            base_spread = 0.3
            if node.is_super_spreader:
                base_spread = 0.6
            if node.forward_label_applied:
                base_spread *= 0.4

            spread_prob = base_spread * group.spread_multiplier
            spread_prob = min(spread_prob, 0.95)

            for conn_id in node.connections:
                conn = self.nodes[conn_id]
                if conn.status != "clean":
                    continue
                skepticism = group.skepticism
                effective_prob = spread_prob * (1.0 - skepticism)
                if self.rng.random() < effective_prob:
                    newly_infected.append((conn_id, node.message_id))

        for node_id, message_id in newly_infected:
            self.nodes[node_id].has_misinfo = True
            self.nodes[node_id].status = "infected"
            self.nodes[node_id].message_id = message_id

        self.time_elapsed += 1
        return len(newly_infected)

    def intercept(self, node_id: int) -> bool:
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        if node.status == "blocked":
            return False
        node.status = "blocked"
        node.has_misinfo = False
        if node.is_super_spreader:
            self.super_spreaders_found += 1
        if node.is_source:
            self.sources_found += 1
        return True

    def quarantine(self, group_id: int) -> int:
        if group_id not in self.groups:
            return 0
        group = self.groups[group_id]
        group.is_quarantined = True
        blocked_count = 0
        for member_id in group.member_ids:
            node = self.nodes[member_id]
            if node.status == "infected":
                node.status = "blocked"
                node.has_misinfo = False
                if node.is_super_spreader:
                    self.super_spreaders_found += 1
                if node.is_source:
                    self.sources_found += 1
                blocked_count += 1
        return blocked_count

    def identify_spreader(self) -> Optional[Tuple[int, str]]:
        best_node = None
        best_score = -1
        for node_id, node in self.nodes.items():
            if node.status != "infected":
                continue
            score = len(node.connections)
            if node.is_super_spreader:
                score *= 2
            if score > best_score:
                best_score = score
                best_node = node
        if best_node is None:
            return None
        return (best_node.id, best_node.message_id or "unknown")

    def add_forward_label(self, node_id: int) -> bool:
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        if node.forward_label_applied:
            return False
        node.forward_label_applied = True
        return True

    def get_infected_count(self) -> int:
        return sum(
            1 for n in self.nodes.values()
            if n.status == "infected"
        )

    def get_state(self) -> NetworkState:
        infected = self.get_infected_count()
        blocked = sum(
            1 for n in self.nodes.values()
            if n.status == "blocked"
        )
        clean = self.total_users - infected - blocked
        forward_labels = sum(
            1 for n in self.nodes.values()
            if n.forward_label_applied
        )
        return NetworkState(
            total_users=self.total_users,
            infected=infected,
            clean=clean,
            blocked=blocked,
            super_spreaders_found=self.super_spreaders_found,
            total_super_spreaders=self.num_super_spreaders,
            sources_found=self.sources_found,
            total_sources=self.num_sources,
            time_elapsed=self.time_elapsed,
            max_time=self.max_time,
            forward_labels_applied=forward_labels
        )