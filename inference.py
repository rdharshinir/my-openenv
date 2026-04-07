import os
import json
import time

from openai import OpenAI
from client import PathosEnv, PathosAction


API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-url>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Initialize OpenAI client with configured variables
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

SYSTEM_PROMPT = """You are a Pathos Rescue Drone navigating a disaster zone.
You must reach the extraction zone while avoiding hazards and moving fires.
Your output must be EXCLUSIVELY a JSON object with two keys:
1. 'reasoning': A short explanation of your logic (max 2 sentences).
2. 'action': Your chosen action. MUST be one of: 'up', 'down', 'left', 'right'.
"""

def play_episode():
    # Stdout logs follow the required structured format (START/STEP/END) exactly
    print("START", flush=True)

    # Connect to local environment server (make sure `uvicorn server.app:app` is running)
    with PathosEnv(base_url="http://localhost:8000").sync() as env:
        # Reset and get initial structured observation
        result = env.reset()
        
        while not result.done:
            print("STEP", flush=True)
            obs = result.observation.structured
            
            # Format the state cleanly for the LLM
            prompt = f"""
Current State:
- Zone Size: {obs['zone_size']}x{obs['zone_size']}
- Disaster Type: {obs['disaster_type']}
- Drone Position: {obs['drone_position']}
- Extraction Zone: {obs['extraction_zone']}
- Distance to extraction: {obs['manhattan_dist_to_extraction']}
- Direction to extraction: {obs['direction_to_extraction']}

Valid Flight Paths (those not immediately resulting in a crash):
"""
            for a in obs['valid_flight_paths']:
                 prompt += f" - {a['label']} -> flies to {a['leads_to']}\n"
                 
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                )
                
                # Parse JSON output from the model
                content = response.choices[0].message.content
                # Strip markdown codeblocks if model wraps output
                if content.startswith("```json"):
                    content = content[7:-3]
                    
                resp_json = json.loads(content)
                reasoning = resp_json.get("reasoning", "")
                action_str = resp_json.get("action", "up")
                
                result = env.step(PathosAction(message=action_str))
                time.sleep(0.5)
                
            except Exception as e:
                # Fallback to safe valid action
                safe_act = "up"
                if obs['valid_flight_paths']:
                    safe_act = obs['valid_flight_paths'][0]['label']
                result = env.step(PathosAction(message=safe_act))
        
    print("END", flush=True)

if __name__ == "__main__":
    play_episode()
