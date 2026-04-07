def grade_episode(trajectory):
    """
    Grade a single Rescue Drone episode.
    """
    total_reward = sum(r for _, _, r, _ in trajectory)
    steps = len(trajectory)

    success = len(trajectory) > 0 and trajectory[-1][2] >= 10.0

    if success:
        baseline_steps = max(20, steps) 
        efficiency = max(0.0, 1.0 - (steps - 1) / baseline_steps)
        score = round(0.5 + 0.5 * efficiency, 4)   
    else:
        final_dist = 0
        grid_size = 5 
        
        if len(trajectory) > 0:
            last_info = trajectory[-1][3] if len(trajectory[-1]) > 3 else {}
            if isinstance(last_info, dict):
                grid_size = last_info.get("grid_size", grid_size)
                if "structured" in last_info:
                    final_dist = last_info["structured"].get("manhattan_dist_to_extraction", grid_size)
        
        max_possible_dist = grid_size * 2
        closeness = max(0.0, 1.0 - (final_dist / max_possible_dist))
        score = round(0.4 * closeness, 4)

    # Detect the difficulty level of the task based on the environment size 
    difficulty = "Easy"
    if grid_size >= 11:
        difficulty = "Hard"
    elif grid_size >= 9:
        difficulty = "Medium"

    return {
        "success":      success,
        "score":        score,
        "steps":        steps,
        "total_reward": round(total_reward, 4),
        "difficulty":   difficulty
    }

# Provide structured tasks for OpenEnv Evaluation
tasks = {
    "Rescue_Level_1_Clear": {"init_kwargs": {"episode": 0}},    # 5x5 Clear Field
    "Rescue_Level_2_Sparse": {"init_kwargs": {"episode": 15}},  # 7x7 Sparse Debris
    "Rescue_Level_3_Maze": {"init_kwargs": {"episode": 25}},    # 9x9 Collapsed Building
    "Rescue_Level_4_Adversarial": {"init_kwargs": {"episode": 35}}, # 11x11 Active Fire
}