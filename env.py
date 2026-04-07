import random
from typing import Optional, Dict, List, Any

# ─────────────────────────────────────────────
# Curriculum Difficulty Config
# ─────────────────────────────────────────────
CURRICULUM_LEVELS = {
    1: {"size": 5,  "map_type": "open",       "fog": False, "wind": False, "label": "Rookie"},
    2: {"size": 7,  "map_type": "sparse",      "fog": False, "wind": False, "label": "Trained"},
    3: {"size": 10, "map_type": "maze",        "fog": True,  "wind": False, "label": "Expert"},
    4: {"size": 10, "map_type": "adversarial", "fog": True,  "wind": True,  "label": "Legendary"},
}

DIFFICULTY_SCHEDULE = [
    (0,   5,  "open"),
    (10,  7,  "sparse"),
    (20,  9,  "maze"),
    (30, 11,  "adversarial"),
]

MAP_TYPES = ("open", "sparse", "maze", "adversarial")


def grid_size_for_episode(episode: int) -> tuple:
    size, mtype = DIFFICULTY_SCHEDULE[0][1], DIFFICULTY_SCHEDULE[0][2]
    for threshold, s, mt in DIFFICULTY_SCHEDULE:
        if episode >= threshold:
            size, mtype = s, mt
    return size, mtype


# ─────────────────────────────────────────────
# Maze Generator (recursive backtracker)
# ─────────────────────────────────────────────

def _generate_maze(size: int, rng: random.Random) -> List[List[bool]]:
    h = w = size
    walls = [[True] * w for _ in range(h)]

    def carve(r, c):
        walls[r][c] = False
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and walls[nr][nc]:
                walls[r + dr // 2][c + dc // 2] = False
                carve(nr, nc)

    start_r = rng.randrange(0, h, 2) if h > 1 else 0
    start_c = rng.randrange(0, w, 2) if w > 1 else 0
    carve(start_r, start_c)
    return walls


class GridEnv:
    """
    Pathos AI – Autonomous Drone Rescue Simulator
    ─────────────────────────────────────────────
    4 Curriculum levels, wind zones, fog of war,
    survivor rescue, speed bonus, replay recording,
    scenario editor support, visit heatmap.

    Actions: 0=up, 1=down, 2=left, 3=right
    """

    STEP_PENALTY  = -0.1
    TRAP_PENALTY  = -10.0
    GOAL_REWARD   = +10.0
    KEY_REWARD    = +0.3
    SPEED_BONUS   = +0.5
    MAX_STEPS: Optional[int] = None

    def __init__(
        self,
        episode: int = 0,
        size: Optional[int] = None,
        map_type: Optional[str] = None,
        seed: Optional[int] = None,
        difficulty: Optional[int] = None,
        custom_layout: Optional[Dict] = None,
    ):
        self.episode = episode
        self.seed = seed
        self._rng = random.Random(seed)
        self.custom_layout = custom_layout

        if difficulty is not None and difficulty in CURRICULUM_LEVELS:
            cfg = CURRICULUM_LEVELS[difficulty]
            self.size = size or cfg["size"]
            self.map_type = map_type or cfg["map_type"]
            self._fog_of_war = cfg["fog"]
            self._wind_enabled = cfg["wind"]
            self.difficulty_label = cfg["label"]
            self.difficulty_level = difficulty
        else:
            sched_size, sched_mtype = grid_size_for_episode(episode)
            self.size = size if size is not None else sched_size
            self.map_type = map_type if map_type is not None else sched_mtype
            self._fog_of_war = self.map_type in ("maze",)
            self._wind_enabled = self.map_type == "adversarial"
            self.difficulty_label = "Custom"
            self.difficulty_level = None

        self._episode_count = 0
        self._walls: List[List[bool]] = []
        self.keys: List[List[int]] = []
        self.keys_collected: int = 0
        self.moving_traps: List[List[int]] = []
        self.traps: List[List[int]] = []
        self.wind_zones: List[List[int]] = []
        self.goal: List[int] = []
        self.agent: List[int] = [0, 0]
        self.steps: int = 0

        # Replay
        self.trajectory: List[Dict] = []
        self.best_trajectory: List[Dict] = []
        self.worst_trajectory: List[Dict] = []
        self._best_score: float = float('-inf')
        self._worst_score: float = float('inf')

        # Heatmap
        self.visit_heatmap: Dict[str, int] = {}

        self.objectives: Dict[str, Any] = {
            "reached_extraction": False,
            "survivors_rescued": 0,
            "speed_bonus": False,
            "total_score": 0.0,
        }

        self.reset()

    # ── Public API ────────────────────────────

    def reset(
        self,
        advance_difficulty: bool = True,
        seed: Optional[int] = None,
        difficulty: Optional[int] = None,
        custom_layout: Optional[Dict] = None,
    ) -> tuple:
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)

        if custom_layout:
            self.custom_layout = custom_layout

        if difficulty is not None and difficulty in CURRICULUM_LEVELS:
            cfg = CURRICULUM_LEVELS[difficulty]
            self.size = cfg["size"]
            self.map_type = cfg["map_type"]
            self._fog_of_war = cfg["fog"]
            self._wind_enabled = cfg["wind"]
            self.difficulty_label = cfg["label"]
            self.difficulty_level = difficulty
        elif advance_difficulty and self._episode_count > 0:
            self._episode_count += 1
            sched_size, sched_mtype = grid_size_for_episode(self.episode + self._episode_count)
            self.size = sched_size
            self.map_type = sched_mtype

        # Archive trajectory
        if self.trajectory:
            ep_score = sum(t["reward"] for t in self.trajectory)
            if ep_score > self._best_score:
                self._best_score = ep_score
                self.best_trajectory = list(self.trajectory)
            if ep_score < self._worst_score:
                self._worst_score = ep_score
                self.worst_trajectory = list(self.trajectory)

        self.trajectory = []
        self.agent = [0, 0]

        if self.custom_layout:
            self._load_custom_layout(self.custom_layout)
        else:
            self._walls = self._build_walls()
            self.goal = self._random_free_cell(exclude=[self.agent])
            self.traps = self._random_traps()
            self.keys = self._random_keys()
            self.moving_traps = self._init_moving_traps()
            self.wind_zones = self._init_wind_zones()

        self.keys_collected = 0
        self.steps = 0
        self.objectives = {
            "reached_extraction": False,
            "survivors_rescued": 0,
            "speed_bonus": False,
            "total_score": 0.0,
        }
        self._record_visit()
        return self._get_state()

    def step(self, action: int) -> tuple:
        x, y = self.agent
        dx, dy = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.get(action, (0, 0))
        nx, ny = x + dx, y + dy

        if not (0 <= nx < self.size and 0 <= ny < self.size) or (
            self._walls and self._walls[nx][ny]
        ):
            nx, ny = x, y

        wind_applied = False
        if self._wind_enabled and [nx, ny] in self.wind_zones:
            wind_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            self._rng.shuffle(wind_dirs)
            for wdr, wdc in wind_dirs:
                wnx, wny = nx + wdr, ny + wdc
                if 0 <= wnx < self.size and 0 <= wny < self.size:
                    if not (self._walls and self._walls[wnx][wny]):
                        nx, ny = wnx, wny
                        wind_applied = True
                        break

        self.agent = [nx, ny]
        self.steps += 1
        self._advance_moving_traps()
        self._record_visit()

        reward = self.STEP_PENALTY
        done = False
        info: Dict[str, Any] = {"wind_applied": wind_applied}

        for k in list(self.keys):
            if self.agent == k:
                self.keys.remove(k)
                self.keys_collected += 1
                reward += self.KEY_REWARD
                self.objectives["survivors_rescued"] = self.keys_collected
                info["survivor_rescued"] = True

        all_traps = self.traps + self.moving_traps
        if self.agent in all_traps:
            reward = self.TRAP_PENALTY
            done = True
            info["result"] = "trap"
        elif self.agent == self.goal:
            reward += self.GOAL_REWARD
            done = True
            info["result"] = "goal"
            self.objectives["reached_extraction"] = True
            if self.steps <= 10:
                reward += self.SPEED_BONUS
                self.objectives["speed_bonus"] = True

        if self.MAX_STEPS and self.steps >= self.MAX_STEPS:
            done = True
            info["result"] = info.get("result", "timeout")

        running_total = sum(t["reward"] for t in self.trajectory) + reward
        self.objectives["total_score"] = round(running_total, 4)

        self.trajectory.append({
            "pos": list(self.agent),
            "action": action,
            "action_label": ["up", "down", "left", "right"][action] if 0 <= action <= 3 else "?",
            "reward": float(reward),
            "done": done,
            "step": self.steps,
            "wind": wind_applied,
        })

        return self._get_state(), reward, done, info

    def render(self) -> str:
        lines = []
        border = "+" + "----+" * self.size
        lines.append(border)
        visible = self._fog_cells() if self._fog_of_war else None
        for r in range(self.size):
            row = "|"
            for c in range(self.size):
                cell = [r, c]
                if visible and (r, c) not in visible:
                    row += " 🌫️ |"
                elif self._walls and self._walls[r][c]:
                    row += " 🧱 |"
                elif cell == self.agent:
                    row += " 🚁 |"
                elif cell == self.goal:
                    row += " 🏥 |"
                elif cell in self.moving_traps:
                    row += " 🔥 |"
                elif cell in self.traps:
                    row += " ☣️ |"
                elif cell in self.keys:
                    row += " 🆘 |"
                elif self._wind_enabled and cell in self.wind_zones:
                    row += " 🌪️ |"
                else:
                    row += " ⬜ |"
            lines.append(row)
            lines.append(border)
        dist = abs(self.agent[0] - self.goal[0]) + abs(self.agent[1] - self.goal[1])
        lines.append(
            f"Steps:{self.steps} | Size:{self.size}x{self.size} | "
            f"Map:{self.map_type} | Drone:{tuple(self.agent)} | "
            f"Extraction:{tuple(self.goal)} | Dist:{dist}"
        )
        return "\n".join(lines)

    def structured_obs(self) -> dict:
        agent_r, agent_c = self.agent
        goal_r, goal_c = self.goal
        manhattan_dist = abs(agent_r - goal_r) + abs(agent_c - goal_c)
        nearby_traps = [
            t for t in self.traps + self.moving_traps
            if abs(t[0] - agent_r) <= 1 and abs(t[1] - agent_c) <= 1 and t != self.agent
        ]
        valid_actions = []
        for act, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            nr, nc = agent_r + dr, agent_c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if not (self._walls and self._walls[nr][nc]):
                    valid_actions.append({
                        "action": act,
                        "label": ["up", "down", "left", "right"][act],
                        "leads_to": [nr, nc],
                        "is_wind_zone": [nr, nc] in self.wind_zones,
                        "is_hazard": [nr, nc] in (self.traps + self.moving_traps),
                    })
        visible_cells = None
        if self._fog_of_war:
            visible_cells = [[r, c] for (r, c) in self._fog_cells()]
        return {
            "drone_position": list(self.agent),
            "extraction_zone": list(self.goal),
            "static_hazards": [list(t) for t in self.traps],
            "active_fires": [list(t) for t in self.moving_traps],
            "wind_zones": [list(w) for w in self.wind_zones],
            "survivors_to_rescue": [list(k) for k in self.keys],
            "survivors_rescued": self.keys_collected,
            "zone_size": self.size,
            "disaster_type": self.map_type,
            "difficulty_level": self.difficulty_level,
            "difficulty_label": self.difficulty_label,
            "battery_used": self.steps,
            "manhattan_dist_to_extraction": manhattan_dist,
            "direction_to_extraction": self._direction_hint(),
            "nearby_danger": len(nearby_traps) > 0,
            "nearby_hazards_coords": nearby_traps,
            "valid_flight_paths": valid_actions,
            "fog_of_war": self._fog_of_war,
            "visible_cells": visible_cells,
            "objectives": dict(self.objectives),
            "rewards": {
                "battery_drain": self.STEP_PENALTY,
                "hazard_penalty": self.TRAP_PENALTY,
                "extraction_reward": self.GOAL_REWARD,
                "survivor_reward": self.KEY_REWARD,
                "speed_bonus": self.SPEED_BONUS,
            },
        }

    def get_grid_for_ui(self) -> dict:
        """Full grid state for SVG/Canvas renderer."""
        visible = self._fog_cells() if self._fog_of_war else None
        cells = []
        for r in range(self.size):
            for c in range(self.size):
                if visible and (r, c) not in visible:
                    ctype = "fog"
                elif self._walls and self._walls[r][c]:
                    ctype = "wall"
                elif [r, c] == self.agent:
                    ctype = "drone"
                elif [r, c] == self.goal:
                    ctype = "extraction"
                elif [r, c] in self.moving_traps:
                    ctype = "fire"
                elif [r, c] in self.traps:
                    ctype = "hazard"
                elif [r, c] in self.keys:
                    ctype = "survivor"
                elif self._wind_enabled and [r, c] in self.wind_zones:
                    ctype = "wind"
                else:
                    ctype = "empty"
                cells.append({"r": r, "c": c, "type": ctype})
        return {
            "cells": cells,
            "size": self.size,
            "drone": list(self.agent),
            "extraction": list(self.goal),
            "static_hazards": [list(t) for t in self.traps],
            "active_fires": [list(t) for t in self.moving_traps],
            "wind_zones": [list(w) for w in self.wind_zones],
            "survivors": [list(k) for k in self.keys],
        }

    def export_layout(self) -> dict:
        """Export map layout as JSON seed for scenario editor."""
        wall_list = [
            [r, c] for r in range(self.size) for c in range(self.size)
            if self._walls and self._walls[r][c]
        ]
        return {
            "size": self.size,
            "map_type": self.map_type,
            "agent": list(self.agent),
            "goal": list(self.goal),
            "traps": [list(t) for t in self.traps],
            "moving_traps": [list(t) for t in self.moving_traps],
            "wind_zones": [list(w) for w in self.wind_zones],
            "keys": [list(k) for k in self.keys],
            "walls": wall_list,
        }

    # ── Internal helpers ───────────────────────

    def _get_state(self) -> tuple:
        return tuple(self.agent)

    def _record_visit(self):
        key = f"{self.agent[0]},{self.agent[1]}"
        self.visit_heatmap[key] = self.visit_heatmap.get(key, 0) + 1

    def _fog_cells(self) -> set:
        ar, ac = self.agent
        visible = set()
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) + abs(dc) <= 2:
                    r, c = ar + dr, ac + dc
                    if 0 <= r < self.size and 0 <= c < self.size:
                        visible.add((r, c))
        return visible

    def _direction_hint(self) -> str:
        dr = self.goal[0] - self.agent[0]
        dc = self.goal[1] - self.agent[1]
        parts = []
        if dr < 0:  parts.append("north")
        elif dr > 0: parts.append("south")
        if dc < 0:  parts.append("west")
        elif dc > 0: parts.append("east")
        return "-".join(parts) if parts else "here"

    def _build_walls(self) -> List[List[bool]]:
        if self.map_type == "maze":
            return _generate_maze(self.size, self._rng)
        return [[False] * self.size for _ in range(self.size)]

    def _random_free_cell(self, exclude=None) -> List[int]:
        exclude = [list(e) for e in (exclude or [])]
        for _ in range(10000):
            r = self._rng.randint(0, self.size - 1)
            c = self._rng.randint(0, self.size - 1)
            cell = [r, c]
            if not (self._walls and self._walls[r][c]) and cell not in exclude:
                return cell
        for r in range(self.size):
            for c in range(self.size):
                cell = [r, c]
                if not (self._walls and self._walls[r][c]) and cell not in exclude:
                    return cell
        return [0, 0]

    def _random_traps(self) -> List[List[int]]:
        rates = {"open": 0.05, "sparse": 0.10, "maze": 0.05, "adversarial": 0.08}
        rate = rates.get(self.map_type, 0.10)
        n_traps = max(1, int(self.size * self.size * rate))
        forbidden = {tuple(self.agent), tuple(self.goal)}
        traps = []
        for _ in range(n_traps * 20):
            if len(traps) >= n_traps:
                break
            t = self._random_free_cell(exclude=list(traps) + [self.agent, self.goal])
            if tuple(t) not in forbidden and t not in traps:
                traps.append(t)
                forbidden.add(tuple(t))
        return traps

    def _random_keys(self) -> List[List[int]]:
        if self.map_type not in ("open", "sparse"):
            return []
        n_keys = 1 if self.size <= 7 else 2
        occupied = [self.agent, self.goal] + self.traps
        keys = []
        for _ in range(n_keys):
            k = self._random_free_cell(exclude=occupied + keys)
            keys.append(k)
        return keys

    def _init_moving_traps(self) -> List[List[int]]:
        if self.map_type not in ("adversarial",):
            return []
        n = max(1, self.size // 4)
        occupied = [self.agent, self.goal] + self.traps
        mtraps = []
        for _ in range(n):
            t = self._random_free_cell(exclude=occupied + mtraps)
            mtraps.append(t)
        return mtraps

    def _init_wind_zones(self) -> List[List[int]]:
        if not self._wind_enabled:
            return []
        n = max(2, self.size // 3)
        occupied = [self.agent, self.goal] + self.traps + self.moving_traps
        zones = []
        for _ in range(n):
            z = self._random_free_cell(exclude=occupied + zones)
            zones.append(z)
        return zones

    def _advance_moving_traps(self):
        for i, trap in enumerate(self.moving_traps):
            options = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = trap[0] + dr, trap[1] + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if not (self._walls and self._walls[nr][nc]):
                        neighbor = [nr, nc]
                        if neighbor != self.goal and neighbor not in self.traps:
                            options.append(neighbor)
            if options:
                self.moving_traps[i] = self._rng.choice(options)

    def _load_custom_layout(self, layout: dict):
        self.size = layout.get("size", self.size)
        self.map_type = layout.get("map_type", self.map_type)
        self.agent = layout.get("agent", [0, 0])
        self.goal = layout.get("goal", [self.size - 1, self.size - 1])
        self.traps = layout.get("traps", [])
        self.moving_traps = layout.get("moving_traps", [])
        self.wind_zones = layout.get("wind_zones", [])
        self.keys = layout.get("keys", [])
        wall_list = layout.get("walls", [])
        self._walls = [[False] * self.size for _ in range(self.size)]
        for r, c in wall_list:
            if 0 <= r < self.size and 0 <= c < self.size:
                self._walls[r][c] = True