"""
Demo Llama 3/4 Agent for the GridMind Benchmark.

This lightweight script connects to the GridMind OpenEnv server, gets the structured
observation, formats it as a prompt, and queries standard Chat Completions API
to get the next move.
"""
import os
import json
import time

# Deferred openai import for when a real API is used

from client import PathosEnv, PathosAction


# Fallback test mock API client if api key is not set
class MockLlamaClient:
    def __init__(self, *args, **kwargs):
        pass
    
    class chat:
        class completions:
            @staticmethod
            def create(model, messages):
                class Msg:
                    @property
                    def content(self):
                        return json.dumps({"reasoning": "Mock reasoning", "action": "south"})
                
                class Choice:
                    def __init__(self):
                        self.message = Msg()

                class Resp:
                    def __init__(self):
                        self.choices = [Choice()]
                return Resp()

API_KEY = os.environ.get("LLAMA_API_KEY", "dummy-key")
BASE_URL = os.environ.get("LLAMA_BASE_URL", "https://api.llama.com/v1")

# Use real OpenAI-compatible client if a key is provided
if API_KEY != "dummy-key":
    try:
        from openai import OpenAI
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    except ImportError:
        print("Please pip install openai to use real models.")
        exit(1)
else:
    print("WARNING: Using mock logic. Set LLAMA_API_KEY environment variable to use real Llama models.")
    client = MockLlamaClient()

SYSTEM_PROMPT = """You are a Pathos Rescue Drone navigating a disaster zone.
You must reach the extraction zone while avoiding hazards and moving fires.
Your output must be EXCLUSIVELY a JSON object with two keys:
1. 'reasoning': A short explanation of your logic (max 2 sentences).
2. 'action': Your chosen action. MUST be one of: 'up', 'down', 'left', 'right'.
"""

def play_episode():
    # Connect to local environment server (make sure `uvicorn server.app:app` is running)
    with PathosEnv(base_url="http://localhost:8000").sync() as env:
        print("Starting Llama Agent Demo...")
        # Reset and get initial structured observation
        result = env.reset()
        
        while not result.done:
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
                    model="llama-3.1-70b-instruct",
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
                
                print(f"🧐 CoT: {reasoning}")
                print(f"🤖 Action: {action_str} -> Map Update:")
                
                result = env.step(PathosAction(message=action_str))
                print(result.observation.grid_state)
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Agent error or JSON parse failure: {e}")
                # Fallback to safe valid action
                safe_act = "up"
                if obs['valid_flight_paths']:
                    safe_act = obs['valid_flight_paths'][0]['label']
                result = env.step(PathosAction(message=safe_act))
        
        print("\n=== Episode Finished ===")
        print(f"Reward: {result.reward}")
        if result.reward >= 10.0:
            print("🎉 Success! Reached Extraction Zone!")
        else:
            print("💀 Failure! Drone was destroyed in a hazard.")
            
if __name__ == "__main__":
    play_episode()
