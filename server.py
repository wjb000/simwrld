#!/usr/bin/env python3
# Python server for MLX model inference - Alternating models for movement

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

@app.route('/', methods=['POST', 'OPTIONS'])
def root_endpoint():
    """Handle requests to the root endpoint for movement decisions"""
    if request.method == 'OPTIONS':
        # Explicitly handle OPTIONS requests for CORS preflight
        return '', 204
    
    data = request.json
    if not data:
        return jsonify({"error": "Missing request data"}), 400
    
    # Extract model_id and data from the request
    model_id = data.get('model_id')
    position = data.get('position', {})
    hit_wall = data.get('hit_wall', False)
    previous_actions = data.get('previous_actions', [])
    current_grid = data.get('current_grid', {})
    
    if not model_id:
        return jsonify({"error": "Missing model_id"}), 400
    
    # Load the model
    model, tokenizer = load_model(model_id)
    if model is None or tokenizer is None:
        return jsonify({"error": f"Failed to load model {model_id}"}), 500
    
    try:
        # Get grid coordinates safely
        grid_x = current_grid.get('x', 0)
        grid_z = current_grid.get('z', 0)
        
        # Create a prompt focused on grid coordinates and previous moves
        movement_prompt = f"""You are an AI controlling a 3D character in a virtual environment. You decide what the character does next based on its current position and previous movements.

CURRENT POSITION: X:{position.get('x', 0):.1f}, Y:{position.get('y', 0):.1f}, Z:{position.get('z', 0):.1f}
CURRENT GRID: {grid_x}, {grid_z}
ENVIRONMENT BOUNDARIES: Walls at x=±50 and z=±50
SPECIAL FEATURES: Wooden platform at (10, 0.5, 10)
{' The character just hit a wall.' if hit_wall else ''}

PREVIOUS ACTIONS: {json.dumps(previous_actions[-5:] if previous_actions else [])}

Based on the character's current grid position and previous movements, decide what the character should do next.

Generate ONE natural, lifelike movement. Return ONLY valid JSON with a single object containing:
- "action": one of [moveForward, moveBackward, moveLeft, moveRight, jump, sprint, idle, explore, lookAround]
- "duration": time in seconds (between 0.5 and 4)
- "thought": a brief description of the character's intention (like "Moving to next grid" or "Exploring current area")

Make your decision interesting and varied. Consider the grid position when planning movements."""
        
        # Format prompt for chat if needed
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": movement_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            formatted_prompt = movement_prompt
        
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
    print("  POST / - Movement decision endpoint (alternating models)")
    print("  POST /check-model - Check if a model can be loaded")
    app.run(host='0.0.0.0', port=5000, debug=True)
