from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
import os
import re
import logging
from llama_cpp import Llama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS with more specific settings
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# Global model cache
models = {}

def load_model(model_id):
    """Load a model if not already loaded"""
    if model_id in models:
        logger.info(f"Using cached model: {model_id}")
        return models[model_id]
    
    logger.info(f"Loading model: {model_id}")
    try:
        # Handle different model paths based on model_id
        if model_id == "mistral7b.gguf":
            model_path = os.path.join(os.path.dirname(__file__), "mistral7b.gguf")
        else:
            # Default to the provided path if it's a full path
            model_path = model_id
            
        logger.info(f"Model path: {model_path}")
        
        # Load the model with llama-cpp-python
        model = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window size
            n_threads=4   # Number of CPU threads to use
        )
        
        models[model_id] = model
        logger.info(f"Successfully loaded {model_id}")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}")
        return None

def extract_json_from_text(text):
    """Extract JSON object from text, or create a fallback JSON if none is found"""
    logger.info(f"Extracting JSON from: {text}")
    
    # Try to find JSON pattern in the text
    json_match = re.search(r'(\{[\s\S]*\})', text)
    
    if json_match:
        try:
            # Try to parse the extracted JSON
            json_obj = json.loads(json_match.group(1))
            logger.info(f"Successfully extracted JSON: {json_obj}")
            return json_obj
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
    
    # If no valid JSON found, create a fallback
    logger.warning("No valid JSON found, using fallback")
    return {
        "action": "explore",
        "duration": 2.5,
        "thought": "Starting to explore the environment"
    }

# Add a route handler for the root path
@app.route('/', methods=['POST', 'OPTIONS'])
def root_endpoint():
    """Handle requests to the root endpoint"""
    if request.method == 'OPTIONS':
        # Explicitly handle OPTIONS requests for CORS preflight
        return '', 204
    
    data = request.json
    if not data:
        logger.warning("Missing request data")
        return jsonify({"error": "Missing request data"}), 400
    
    # Extract model_id and prompt from the request
    model_id = data.get('model_id')
    prompt = data.get('prompt')
    position = data.get('position', {})
    hit_wall = data.get('hit_wall', False)
    previous_actions = data.get('previous_actions', [])
    
    logger.info(f"Request received - Model: {model_id}")
    logger.info(f"Character position: X:{position.get('x', 0):.1f}, Y:{position.get('y', 0):.1f}, Z:{position.get('z', 0):.1f}")
    logger.info(f"Hit wall: {hit_wall}")
    logger.info(f"Previous actions: {previous_actions[-3:] if previous_actions else []}")
    
    if not model_id or not prompt:
        logger.warning("Missing model_id or prompt")
        return jsonify({"error": "Missing model_id or prompt"}), 400
    
    # Load the model
    model = load_model(model_id)
    if model is None:
        logger.error(f"Failed to load model {model_id}")
        return jsonify({"error": f"Failed to load model {model_id}"}), 500
    
    try:
        # Create a simplified prompt that focuses on coordinates and wall collisions
        # Use a more direct prompt that's easier for the model to respond to
        simplified_prompt = f"""You are controlling a 3D character in a virtual environment. Generate the next action for the character.

Current position: X:{position.get('x', 0):.1f}, Y:{position.get('y', 0):.1f}, Z:{position.get('z', 0):.1f}
Environment: Open field with walls at x=±50 and z=±50
{' The character just hit a wall.' if hit_wall else ''}

Choose ONE action from: moveForward, moveBackward, moveLeft, moveRight, jump, sprint, idle, explore, lookAround
Choose a duration between 0.5 and 4 seconds
Include a brief thought describing the character's intention

Format your response EXACTLY like this JSON:
{{"action": "moveForward", "duration": 2.5, "thought": "Exploring the open field"}}

Previous actions: {json.dumps(previous_actions[-3:] if previous_actions else [])}"""

        logger.info("Sending prompt to model:")
        logger.info("-" * 40)
        logger.info(simplified_prompt)
        logger.info("-" * 40)
        
        # Generate response using llama-cpp with higher temperature for more creativity
        response = model.create_completion(
            prompt=simplified_prompt,
            max_tokens=256,
            temperature=0.8,
            top_p=0.95,
            stop=["</s>", "\n\n"],
            echo=False
        )
        
        # Extract the generated text
        generated_text = response["choices"][0]["text"].strip()
        logger.info("Raw model response:")
        logger.info("-" * 40)
        logger.info(generated_text)
        logger.info("-" * 40)
        
        # If the model didn't generate anything, use a predefined action
        if not generated_text:
            logger.warning("Model returned empty response, using predefined action")
            # Cycle through different actions to make the character move
            if not previous_actions:
                action = "lookAround"
                thought = "Taking in the surroundings"
            elif len(previous_actions) == 1:
                action = "moveForward"
                thought = "Starting to explore the environment"
            elif len(previous_actions) == 2:
                action = "moveRight"
                thought = "Checking what's to the right"
            elif len(previous_actions) == 3:
                action = "sprint"
                thought = "Moving quickly to cover more ground"
            else:
                action = "explore"
                thought = "Continuing to explore the area"
                
            json_response = {
                "action": action,
                "duration": 2.0,
                "thought": thought
            }
        else:
            # Extract JSON from the response or create fallback
            json_response = extract_json_from_text(generated_text)
        
        # Ensure the response has all required fields
        if "action" not in json_response:
            logger.warning("Missing 'action' field, using fallback")
            json_response["action"] = "explore"
        if "duration" not in json_response:
            logger.warning("Missing 'duration' field, using fallback")
            json_response["duration"] = 2.0
        if "thought" not in json_response:
            logger.warning("Missing 'thought' field, using fallback")
            json_response["thought"] = "Exploring the environment"
            
        # Validate action is in allowed list
        allowed_actions = ["moveForward", "moveBackward", "moveLeft", "moveRight", 
                          "jump", "sprint", "idle", "explore", "lookAround"]
        if json_response["action"] not in allowed_actions:
            logger.warning(f"Invalid action '{json_response['action']}', using fallback")
            json_response["action"] = "explore"
            
        # Validate duration is within reasonable range
        if not isinstance(json_response["duration"], (int, float)) or json_response["duration"] < 0.5 or json_response["duration"] > 4:
            logger.warning(f"Invalid duration '{json_response['duration']}', using fallback")
            json_response["duration"] = 2.0
        
        # Convert to JSON string
        json_string = json.dumps(json_response)
        logger.info(f"Final response: {json_string}")
            
        # Return the JSON string directly
        return jsonify({"response": json_string})
    
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        # Return a fallback response on error
        fallback = {"action": "explore", "duration": 2.0, "thought": "Exploring the environment"}
        return jsonify({"response": json.dumps(fallback)}), 200

@app.route('/generate', methods=['POST'])
def generate_text():
    """Generate text from a model"""
    data = request.json
    model_id = data.get('model_id')
    prompt = data.get('prompt')
    
    logger.info(f"Generate text request - Model: {model_id}")
    
    if not model_id or not prompt:
        logger.warning("Missing model_id or prompt")
        return jsonify({"error": "Missing model_id or prompt"}), 400
    
    # Load the model
    model = load_model(model_id)
    if model is None:
        logger.error(f"Failed to load model {model_id}")
        return jsonify({"error": f"Failed to load model {model_id}"}), 500
    
    try:
        logger.info("Prompt:")
        logger.info("-" * 40)
        logger.info(prompt)
        logger.info("-" * 40)
        
        # Generate response using llama-cpp
        response = model.create_completion(
            prompt=prompt,
            max_tokens=256,
            temperature=0.7,
            stop=["</s>", "\n\n"],
            echo=False
        )
        
        # Extract the generated text
        generated_text = response["choices"][0]["text"]
        
        logger.info("Response:")
        logger.info("-" * 40)
        logger.info(generated_text)
        logger.info("-" * 40)
        
        return jsonify({"response": generated_text})
    
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Fix: Change the route to match what the client is expecting
@app.route('/check-model', methods=['POST'])
def check_model():
    """Check if a model can be loaded"""
    data = request.json
    model_id = data.get('model_id')
    
    logger.info(f"Check model request: {model_id}")
    
    if not model_id:
        logger.warning("Missing model_id")
        return jsonify({"error": "Missing model_id"}), 400
    
    # Try to load the model
    model = load_model(model_id)
    if model is None:
        logger.warning(f"Failed to load model {model_id}")
        return jsonify({"success": False, "error": f"Failed to load model {model_id}"}), 200
    
    logger.info(f"Model {model_id} loaded successfully")
    return jsonify({"success": True, "model_id": model_id}), 200

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Starting LLaMA model server on http://localhost:5000")
    logger.info("Available endpoints:")
    logger.info("  POST /generate - Generate text from a model")
    logger.info("  POST /check-model - Check if a model can be loaded")
    logger.info("  POST / - Root endpoint for basic requests")
    logger.info("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
