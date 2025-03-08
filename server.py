#!/usr/bin/env python3
# Python server for MLX model inference - Tag Game with LLM-controlled characters

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

@app.route('/tag-game', methods=['POST', 'OPTIONS'])
def tag_game_endpoint():
    """Handle requests for tag game movement decisions"""
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
    opponent_position = data.get('opponent_position', {})
    opponent_grid = data.get('opponent_grid', {})
    is_chaser = data.get('is_chaser', False)
    tag_count = data.get('tag_count', 0)
    distance_to_opponent = data.get('distance_to_opponent', 999)
    direction_to_opponent = data.get('direction_to_opponent', {"x": 0, "z": 0})
    
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
        
        # Create a prompt based on the character's role (chaser or runner)
        role = "CHASER" if is_chaser else "RUNNER"
        
        if is_chaser:
            movement_prompt = f"""You are an AI controlling a character in a tag game. Your role is CHASER. You need to catch the opponent by moving close to them.

CURRENT POSITION: X:{position.get('x', 0):.1f}, Y:{position.get('y', 0):.1f}, Z:{position.get('z', 0):.1f}
CURRENT GRID: {grid_x}, {grid_z}
ROLE: CHASER (you need to catch the opponent)
TAG COUNT: {tag_count}

OPPONENT POSITION: X:{opponent_position.get('x', 0):.1f}, Y:{opponent_position.get('y', 0):.1f}, Z:{opponent_position.get('z', 0):.1f}
OPPONENT GRID: {opponent_grid.get('x', 0)}, {opponent_grid.get('z', 0)}
DISTANCE TO OPPONENT: {distance_to_opponent:.1f} units

ENVIRONMENT BOUNDARIES: Walls at x=±50 and z=±50
SPECIAL FEATURES: Wooden platform at (10, 0.5, 10)
{' You just hit a wall.' if hit_wall else ''}

PREVIOUS ACTIONS: {json.dumps(previous_actions[-3:] if previous_actions else [])}

CHASER STRATEGY TIPS:
1. Move toward the opponent to tag them
2. Use sprint when you're close to catch up
3. Try to predict where they're going and intercept
4. Use diagonal movements to move faster
5. If the opponent is hiding, explore the area

Choose ONE movement action. Return ONLY valid JSON with a single object containing:
- "action": one of [moveForward, moveBackward, moveLeft, moveRight, moveDiagonalForwardLeft, moveDiagonalForwardRight, moveDiagonalBackwardLeft, moveDiagonalBackwardRight, jump, sprint, idle, explore, lookAround]
- "duration": time in seconds (between 0.5 and 3)
- "thought": a brief description of your strategy (like "Chasing opponent" or "Moving to intercept")"""
        else:
            movement_prompt = f"""You are an AI controlling a character in a tag game. Your role is RUNNER. You need to avoid being caught by the opponent.

CURRENT POSITION: X:{position.get('x', 0):.1f}, Y:{position.get('y', 0):.1f}, Z:{position.get('z', 0):.1f}
CURRENT GRID: {grid_x}, {grid_z}
ROLE: RUNNER (you need to avoid being caught)
TAG COUNT: {tag_count}

OPPONENT POSITION: X:{opponent_position.get('x', 0):.1f}, Y:{opponent_position.get('y', 0):.1f}, Z:{opponent_position.get('z', 0):.1f}
OPPONENT GRID: {opponent_grid.get('x', 0)}, {opponent_grid.get('z', 0)}
DISTANCE TO OPPONENT: {distance_to_opponent:.1f} units

ENVIRONMENT BOUNDARIES: Walls at x=±50 and z=±50
SPECIAL FEATURES: Wooden platform at (10, 0.5, 10)
{' You just hit a wall.' if hit_wall else ''}

PREVIOUS ACTIONS: {json.dumps(previous_actions[-3:] if previous_actions else [])}

RUNNER STRATEGY TIPS:
1. Move away from the opponent to avoid being tagged
2. Use sprint when the opponent is close to escape
3. Change direction frequently to be unpredictable
4. Use diagonal movements to move faster
5. Use the environment to your advantage (like hiding behind obstacles)

Choose ONE movement action. Return ONLY valid JSON with a single object containing:
- "action": one of [moveForward, moveBackward, moveLeft, moveRight, moveDiagonalForwardLeft, moveDiagonalForwardRight, moveDiagonalBackwardLeft, moveDiagonalBackwardRight, jump, sprint, idle, explore, lookAround]
- "duration": time in seconds (between 0.5 and 3)
- "thought": a brief description of your strategy (like "Evading chaser" or "Finding hiding spot")"""
        
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
    print("Starting MLX model server for Tag Game on http://localhost:5000")
    print("Available endpoints:")
    print("  POST /tag-game - Tag game movement decision endpoint")
    print("  POST /check-model - Check if a model can be loaded")
    app.run(host='0.0.0.0', port=5000, debug=True)
