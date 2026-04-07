def grade_episode(trajectory):
    """
    Grade a single Pathos AI Drone episode.

    Scoring:
      Success  → 0.5 + 0.5 * efficiency  (efficiency based on step count)
      Speed    → +0.1 bonus if completed in ≤10 steps
      Survivor → +0.05 per survivor rescued (capped at +0.1)
      Failure  → 0.0–0.4 based on proximity to extraction zone
    """
    if not trajectory:
        return {"success": False, "score": 0.0, "steps": 0,
                "total_reward": 0.0, "difficulty": "Unknown"}

    total_reward = sum(r for _, _, r, _ in trajectory)
    steps = len(trajectory)

    # Detect success: last step reward >= 10 (GOAL_REWARD)
    last_reward = trajectory[-1][2]
    success = last_reward >= 10.0

    # Detect speed bonus
    speed_bonus = success and steps <= 10

    # Count survivors rescued (steps with reward spike of +0.3)
    survivors_rescued = sum(
        1 for _, _, r, _ in trajectory
        if 0.2 <= r <= 0.5
    )

    # Detect difficulty from info dict
    grid_size = 5
    final_dist = 0
    if trajectory[-1][3] and isinstance(trajectory[-1][3], dict):
        info = trajectory[-1][3]
        grid_size = info.get("grid_size", grid_size)
        structured = info.get("structured", {})
        if structured:
            grid_size = structured.get("zone_size", grid_size)
            final_dist = structured.get("manhattan_dist_to_extraction", grid_size)

    if success:
        baseline = max(20, steps)
        efficiency = max(0.0, 1.0 - (steps - 1) / baseline)
        score = 0.5 + 0.5 * efficiency
        if speed_bonus:
            score += 0.1
        score += min(0.1, survivors_rescued * 0.05)
        score = min(1.0, round(score, 4))
    else:
        max_dist = grid_size * 2
        closeness = max(0.0, 1.0 - (final_dist / max(1, max_dist)))
        score = round(0.4 * closeness + survivors_rescued * 0.05, 4)

    # Difficulty label
    if grid_size >= 10:
        difficulty = "Legendary" if grid_size >= 11 else "Expert"
    elif grid_size >= 7:
        difficulty = "Trained"
    else:
        difficulty = "Rookie"

    return {
        "success":         success,
        "score":           score,
        "steps":           steps,
        "total_reward":    round(total_reward, 4),
        "difficulty":      difficulty,
        "speed_bonus":     speed_bonus,
        "survivors_rescued": survivors_rescued,
    }


# Structured tasks for OpenEnv Evaluation
tasks = {
    "Rescue_L1_Rookie":     {"init_kwargs": {"episode": 0,  "difficulty": 1}},
    "Rescue_L2_Trained":    {"init_kwargs": {"episode": 15, "difficulty": 2}},
    "Rescue_L3_Expert":     {"init_kwargs": {"episode": 25, "difficulty": 3}},
    "Rescue_L4_Legendary":  {"init_kwargs": {"episode": 35, "difficulty": 4}},
}