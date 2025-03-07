#!/usr/bin/env python3
# Python server for MLX model inference

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
import os
from mlx_lm import load, generate

app = Flask(__name__)
# Configure CORS with more specific settings
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# Global model cache
models = {}

def load_model(model_id):
    """Load a model if not already loaded"""
    if model_id in models:
        return models[model_id]
    
    print(f"Loading model: {model_id}")
    try:
        model, tokenizer = load(model_id)
        models[model_id] = (model, tokenizer)
        print(f"Successfully loaded {model_id}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None, None

# Add a route handler for the root path
@app.route('/', methods=['POST', 'OPTIONS'])
def root_endpoint():
    """Handle requests to the root endpoint"""
    if request.method == 'OPTIONS':
        # Explicitly handle OPTIONS requests for CORS preflight
        return '', 204
    
    data = request.json
    if not data:
        return jsonify({"error": "Missing request data"}), 400
    
    # Extract model_id and prompt from the request
    model_id = data.get('model_id')
    prompt = data.get('prompt')
    position = data.get('position', {})
    hit_wall = data.get('hit_wall', False)
    previous_actions = data.get('previous_actions', [])
    
    if not model_id or not prompt:
        return jsonify({"error": "Missing model_id or prompt"}), 400
    
    # Load the model
    model, tokenizer = load_model(model_id)
    if model is None or tokenizer is None:
        return jsonify({"error": f"Failed to load model {model_id}"}), 500
    
    try:
        # Create a simplified prompt that focuses on coordinates and wall collisions
        simplified_prompt = f"""You are an AI controlling a 3D character in a virtual environment. You decide what the character does next based on its current state and surroundings. The environment has walls at the boundaries (±50 on x and z axes).

The character is at position X:{position.get('x', 0):.1f}, Y:{position.get('y', 0):.1f}, Z:{position.get('z', 0):.1f} in an open field. The environment has walls at x=±50 and z=±50.
{' The character just hit a wall.' if hit_wall else ''}

Generate ONE natural, lifelike movement for the character to perform next. Return ONLY valid JSON with a single object containing:
- "action": one of [moveForward, moveBackward, moveLeft, moveRight, jump, sprint, idle, explore, lookAround]
- "duration": time in seconds (between 0.5 and 4)
- "thought": a brief description of the character's intention (like "Exploring the area" or "Getting tired, need to rest")

Create an action that would make sense for a character exploring this environment. Make the movement feel natural and purposeful. Avoid actions that would cause the character to hit walls.

Previous actions: {json.dumps(previous_actions[-3:] if previous_actions else [])}"""
        
        # Format prompt for chat if needed
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": simplified_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            formatted_prompt = simplified_prompt
        
        # Generate response
        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=256,
            verbose=False
        )
        
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_text():
    """Generate text from a model"""
    data = request.json
    model_id = data.get('model_id')
    prompt = data.get('prompt')
    
    if not model_id or not prompt:
        return jsonify({"error": "Missing model_id or prompt"}), 400
    
    # Load the model
    model, tokenizer = load_model(model_id)
    if model is None or tokenizer is None:
        return jsonify({"error": f"Failed to load model {model_id}"}), 500
    
    try:
        # Format prompt for chat if needed
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        # Generate response
        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=256,
            verbose=False
        )
        
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": str(e)}), 500

# Fix: Change the route to match what the client is expecting
@app.route('/check-model', methods=['POST'])
def check_model():
    """Check if a model can be loaded"""
    data = request.json
    model_id = data.get('model_id')
    
    if not model_id:
        return jsonify({"error": "Missing model_id"}), 400
    
    # Try to load the model
    model, tokenizer = load_model(model_id)
    if model is None or tokenizer is None:
        return jsonify({"success": False, "error": f"Failed to load model {model_id}"}), 200
    
    return jsonify({"success": True, "model_id": model_id}), 200

if __name__ == '__main__':
    print("Starting MLX model server on http://localhost:5000")
    # Add more verbose output to help with debugging
    print("Available endpoints:")
    print("  POST /generate - Generate text from a model")
    print("  POST /check-model - Check if a model can be loaded")
    print("  POST / - Root endpoint for basic requests")
    app.run(host='0.0.0.0', port=5000, debug=True)