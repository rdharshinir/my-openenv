import random
import math
from collections import deque
from typing import Optional


# ─────────────────────────────────────────────
# Difficulty schedule: episode → (grid size, map type)
# ─────────────────────────────────────────────
DIFFICULTY_SCHEDULE = [
    (0,   5,  "open"),         # episodes  0–9   → 5×5  open field
    (10,  7,  "sparse"),       # episodes 10–19  → 7×7  sparse traps
    (20,  9,  "maze"),         # episodes 20–29  → 9×9  maze corridors
    (30, 11,  "adversarial"),  # episodes 30+    → 11×11 moving traps
]

MAP_TYPES = ("open", "sparse", "maze", "adversarial")


def grid_size_for_episode(episode: int) -> tuple[int, str]:
    """Return (grid_size, map_type) for a given episode index."""
    size, mtype = DIFFICULTY_SCHEDULE[0][1], DIFFICULTY_SCHEDULE[0][2]
    for threshold, s, mt in DIFFICULTY_SCHEDULE:
        if episode >= threshold:
            size, mtype = s, mt
    return size, mtype


# ─────────────────────────────────────────────
# Maze Generator (recursive backtracker)
# ─────────────────────────────────────────────

def _generate_maze(size: int, rng: random.Random) -> list[list[bool]]:
    """
    Returns a grid of booleans: True = wall, False = passable.
    Guaranteed to have a connected open space using recursive backtracking.
    Works on odd sizes; even sizes are padded.
    """
    # Work on a half-resolution grid, then expand
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
    Pathos AI Drone Rescue Simulator:
      - Navigate as an autonomous drone 🚁
      - Rescue stranded survivors 🆘 before reaching extraction 🏥
      - Avoid static hazards ☣️ and active spreading fires 🔥
      - Evaluates spatial mapping and risk mitigation in LLMs

    Actions: 0=up, 1=down, 2=left, 3=right
    """

    # ── Reward shaping knobs ──
    STEP_PENALTY   = -0.1    # battery drain per step
    TRAP_PENALTY   = -10.0   # flying into a hazard/fire (fatal)
    GOAL_REWARD    = +10.0   # reaching extraction safety
    KEY_REWARD     = +5.0    # rescuing a survivor
    MAX_STEPS      = None    # maximum safe flight time

    def __init__(
        self,
        episode: int = 0,
        size: Optional[int] = None,
        map_type: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            episode:  used to look up grid size & map type from the difficulty schedule.
            size:     override the grid size directly (ignores episode schedule).
            map_type: override the map type ("open", "sparse", "maze", "adversarial").
            seed:     RNG seed for reproducible environments.
        """
        self.episode = episode
        self.seed = seed
        self._rng = random.Random(seed)

        sched_size, sched_mtype = grid_size_for_episode(episode)
        self.size = size if size is not None else sched_size
        self.map_type = map_type if map_type is not None else sched_mtype

        self._episode_count = 0
        self._walls: list[list[bool]] = []
        self.keys: list[list] = []      # collectible keys
        self.keys_collected: int = 0
        self.moving_traps: list[list] = []  # adversarial moving traps
        self.traps: list[list] = []
        self.goal: list = []
        self.agent: list = [0, 0]
        self.steps: int = 0
        self.reset()

    # ── Public API ────────────────────────────

    def reset(self, advance_difficulty: bool = True, seed: Optional[int] = None) -> tuple:
        """
        Reset the environment.

        Args:
            advance_difficulty: if True, increment the internal episode counter
                                and potentially grow the grid.
            seed: override the RNG seed for this episode.
        Returns:
            Initial state tuple (agent_row, agent_col).
        """
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)

        if advance_difficulty and self._episode_count > 0:
            self._episode_count += 1
            sched_size, sched_mtype = grid_size_for_episode(self.episode + self._episode_count)
            self.size = sched_size
            self.map_type = sched_mtype

        self.agent = [0, 0]
        self._walls = self._build_walls()
        self.goal = self._random_free_cell(exclude=[self.agent])
        self.traps = self._random_traps()
        self.keys = self._random_keys()
        self.moving_traps = self._init_moving_traps()
        self.keys_collected = 0
        self.steps = 0
        return self._get_state()

    def step(self, action: int) -> tuple:
        """
        Execute one step.

        Args:
            action: int in {0,1,2,3}
        Returns:
            (state, reward, done, info)
        """
        x, y = self.agent

        dx, dy = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.get(action, (0, 0))
        nx, ny = x + dx, y + dy

        # Wall / boundary collision
        if not (0 <= nx < self.size and 0 <= ny < self.size) or (
            self._walls and self._walls[nx][ny]
        ):
            nx, ny = x, y  # stay in place, still pay step penalty

        self.agent = [nx, ny]
        self.steps += 1

        # Advance moving traps
        self._advance_moving_traps()

        reward = self.STEP_PENALTY
        done = False
        info: dict = {}

        # Check key pickup
        for k in list(self.keys):
            if self.agent == k:
                self.keys.remove(k)
                self.keys_collected += 1
                reward += self.KEY_REWARD
                info["key_picked"] = True

        # Check trap collision (static + moving)
        all_traps = self.traps + self.moving_traps
        if self.agent in all_traps:
            reward = self.TRAP_PENALTY
            done = True
            info["result"] = "trap"

        elif self.agent == self.goal:
            reward = self.GOAL_REWARD
            done = True
            info["result"] = "goal"

        if self.MAX_STEPS and self.steps >= self.MAX_STEPS:
            done = True
            info["result"] = info.get("result", "timeout")

        return self._get_state(), reward, done, info

    def render(self) -> str:
        """
        Return a beautiful Emoji representation of the disaster zone.

        Legend:
            🚁 – Drone (Agent)
            🏥 – Extraction (Goal)
            ☣️ – Static Hazard (Trap)
            🔥 – Spreading Fire (Moving Trap)
            🆘 – Survivor (Key)
            🧱 – Debris (Wall)
            ⬜ – Clear path
        """
        lines = []
        # Adjusted border for double-wide emojis
        border = "+" + "----+" * self.size
        lines.append(border)

        for r in range(self.size):
            row = "|"
            for c in range(self.size):
                cell = [r, c]
                if self._walls and self._walls[r][c]:
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
                else:
                    row += " ⬜ |"
            lines.append(row)
            lines.append(border)

        agent_pos = tuple(self.agent)
        goal_pos = tuple(self.goal)
        dist = abs(self.agent[0] - self.goal[0]) + abs(self.agent[1] - self.goal[1])
        lines.append(
            f"Battery Steps: {self.steps}  |  Zone Size: {self.size}×{self.size}  "
            f"|  Map: {self.map_type}  |  Drone: {agent_pos}  "
            f"|  Extraction: {goal_pos}  |  Dist: {dist}"
        )
        return "\n".join(lines)

    def structured_obs(self) -> dict:
        """
        Return a rich structured observation dict suitable for LLM consumption.
        """
        agent_r, agent_c = self.agent
        goal_r, goal_c = self.goal

        manhattan_dist = abs(agent_r - goal_r) + abs(agent_c - goal_c)
        direction_hint = self._direction_hint()

        # Nearby danger: any trap in 1-step radius
        nearby_traps = [
            t for t in self.traps + self.moving_traps
            if abs(t[0] - agent_r) <= 1 and abs(t[1] - agent_c) <= 1 and t != self.agent
        ]

        # Valid actions (not hitting boundary or wall)
        valid_actions = []
        for act, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            nr, nc = agent_r + dr, agent_c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if not (self._walls and self._walls[nr][nc]):
                    valid_actions.append({
                        "action": act,
                        "label": ["up", "down", "left", "right"][act],
                        "leads_to": [nr, nc],
                    })

        return {
            "drone_position": list(self.agent),
            "extraction_zone": list(self.goal),
            "static_hazards": [list(t) for t in self.traps],
            "active_fires": [list(t) for t in self.moving_traps],
            "survivors_to_rescue": [list(k) for k in self.keys],
            "survivors_rescued": self.keys_collected,
            "zone_size": self.size,
            "disaster_type": self.map_type,
            "battery_used": self.steps,
            "manhattan_dist_to_extraction": manhattan_dist,
            "direction_to_extraction": direction_hint,
            "nearby_danger": len(nearby_traps) > 0,
            "nearby_hazards_coords": nearby_traps,
            "valid_flight_paths": valid_actions,
            "rewards": {
                "battery_drain": self.STEP_PENALTY,
                "hazard_penalty": self.TRAP_PENALTY,
                "extraction_reward": self.GOAL_REWARD,
                "survivor_reward": self.KEY_REWARD,
            },
        }

    # ── Internal helpers ───────────────────────

    def _get_state(self) -> tuple:
        return tuple(self.agent)

    def _direction_hint(self) -> str:
        """Human-readable direction from agent to goal."""
        dr = self.goal[0] - self.agent[0]
        dc = self.goal[1] - self.agent[1]
        parts = []
        if dr < 0:
            parts.append("north")
        elif dr > 0:
            parts.append("south")
        if dc < 0:
            parts.append("west")
        elif dc > 0:
            parts.append("east")
        return "-".join(parts) if parts else "here"

    def _build_walls(self) -> list[list[bool]]:
        """Build wall map based on map_type."""
        if self.map_type == "maze":
            return _generate_maze(self.size, self._rng)
        # For all other modes, no structural walls (traps serve as obstacles)
        return [[False] * self.size for _ in range(self.size)]

    def _random_free_cell(self, exclude: list | None = None) -> list:
        """Pick a random cell that is passable (not a wall) and not in exclude."""
        exclude = [list(e) for e in (exclude or [])]
        for _ in range(10000):
            r = self._rng.randint(0, self.size - 1)
            c = self._rng.randint(0, self.size - 1)
            cell = [r, c]
            if not (self._walls and self._walls[r][c]) and cell not in exclude:
                return cell
        # Fallback: scan deterministically
        for r in range(self.size):
            for c in range(self.size):
                cell = [r, c]
                if not (self._walls and self._walls[r][c]) and cell not in exclude:
                    return cell
        return [0, 0]

    def _random_goal(self) -> list:
        """Pick a goal that is not the agent's start position."""
        return self._random_free_cell(exclude=[self.agent])

    def _random_traps(self) -> list[list]:
        """
        Scatter traps based on map_type:
        - open:        ~5% of cells
        - sparse:      ~10% of cells
        - maze:        ~5% (walls already block movement)
        - adversarial: ~8% static traps (moving traps added separately)
        """
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

    def _random_keys(self) -> list[list]:
        """Add collectible keys only in sparse/open modes."""
        if self.map_type not in ("open", "sparse"):
            return []
        n_keys = 1 if self.size <= 7 else 2
        occupied = [self.agent, self.goal] + self.traps
        keys = []
        for _ in range(n_keys):
            k = self._random_free_cell(exclude=occupied + keys)
            keys.append(k)
        return keys

    def _init_moving_traps(self) -> list[list]:
        """Initialize moving traps for adversarial mode."""
        if self.map_type != "adversarial":
            return []
        n = max(1, self.size // 4)
        occupied = [self.agent, self.goal] + self.traps
        mtraps = []
        for _ in range(n):
            t = self._random_free_cell(exclude=occupied + mtraps)
            mtraps.append(t)
        return mtraps

    def _advance_moving_traps(self):
        """Move each adversarial trap one step randomly (avoid boundaries/walls)."""
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