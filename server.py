#!/usr/bin/env python3
# Python server for MLX model inference - Social Interaction Simulation with LLM-controlled characters

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
import os
import math
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

def calculate_direction_advice(position, other_position):
    """Calculate simple direction advice to help AI understand where to go"""
    dx = other_position.get('x', 0) - position.get('x', 0)
    dz = other_position.get('z', 0) - position.get('z', 0)
    
    # Determine primary direction
    if abs(dx) > abs(dz):
        # X-axis is primary direction
        x_direction = "east" if dx > 0 else "west"
        z_direction = "south" if dz > 0 else "north"
        primary = f"primarily {x_direction}"
        if abs(dz) > 5:  # Only mention secondary if significant
            primary += f" and slightly {z_direction}"
    else:
        # Z-axis is primary direction
        x_direction = "east" if dx > 0 else "west"
        z_direction = "south" if dz > 0 else "north"
        primary = f"primarily {z_direction}"
        if abs(dx) > 5:  # Only mention secondary if significant
            primary += f" and slightly {x_direction}"
    
    return primary

def suggest_movement_action(position, other_position):
    """Suggest a specific movement action based on relative positions"""
    dx = other_position.get('x', 0) - position.get('x', 0)
    dz = other_position.get('z', 0) - position.get('z', 0)
    
    # Normalize to get direction
    distance = math.sqrt(dx*dx + dz*dz)
    if distance < 0.1:  # Avoid division by zero
        return "idle"
    
    # Normalize
    dx = dx / distance
    dz = dz / distance
    
    # Determine movement direction based on the angle
    # Forward is -z, right is +x in the coordinate system
    
    # Check for diagonal movements first
    if dx > 0.5 and dz < -0.5:
        return "move_forward_right"
    elif dx < -0.5 and dz < -0.5:
        return "move_forward_left"
    elif dx > 0.5 and dz > 0.5:
        return "move_backward_right"
    elif dx < -0.5 and dz > 0.5:
        return "move_backward_left"
    # Then check for cardinal directions
    elif abs(dx) > abs(dz):
        return "move_right" if dx > 0 else "move_left"
    else:
        return "move_forward" if dz < 0 else "move_backward"

@app.route('/social-interaction', methods=['POST', 'OPTIONS'])
def social_interaction_endpoint():
    """Handle requests for social interaction decisions"""
    if request.method == 'OPTIONS':
        # Explicitly handle OPTIONS requests for CORS preflight
        return '', 204
    
    data = request.json
    if not data:
        return jsonify({"error": "Missing request data"}), 400
    
    # Extract data from the request
    model_id = data.get('model_id')
    position = data.get('position', {})
    hit_wall = data.get('hit_wall', False)
    previous_actions = data.get('previous_actions', [])
    current_grid = data.get('current_grid', {})
    other_position = data.get('other_position', {})
    other_grid = data.get('other_grid', {})
    distance_to_other = data.get('distance_to_other', 999)
    direction_to_other = data.get('direction_to_other', {"x": 0, "z": 0})
    character_name = data.get('character_name', 'Character')
    other_name = data.get('other_name', 'Other')
    recent_messages = data.get('recent_messages', [])
    
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
        
        # Calculate helpful direction advice
        direction_advice = calculate_direction_advice(position, other_position)
        suggested_action = suggest_movement_action(position, other_position)
        
        # Format recent messages for the prompt
        conversation_history = ""
        if recent_messages:
            conversation_history = "RECENT CONVERSATION:\n"
            for msg in recent_messages[-5:]:  # Use last 5 messages
                speaker = msg.get('speaker', 'Unknown')
                message = msg.get('message', '')
                conversation_history += f"{speaker}: {message}\n"
        
        # Create a prompt for social interaction with emphasis on seeking each other out
        interaction_prompt = f"""You are an AI controlling a character named {character_name} in a social simulation. Your PRIMARY GOAL is to find {other_name} and have friendly conversations. You should actively seek out {other_name} and engage with them.

CURRENT POSITION: X:{position.get('x', 0):.1f}, Y:{position.get('y', 0):.1f}, Z:{position.get('z', 0):.1f}
CURRENT GRID: {grid_x}, {grid_z}
CHARACTER: {character_name}

{other_name}'S POSITION: X:{other_position.get('x', 0):.1f}, Y:{other_position.get('y', 0):.1f}, Z:{other_position.get('z', 0):.1f}
{other_name}'S GRID: {other_grid.get('x', 0)}, {other_grid.get('z', 0)}
DISTANCE TO {other_name}: {distance_to_other:.1f} units

DIRECTION TO {other_name}: {direction_advice}
SUGGESTED MOVEMENT: {suggested_action}

ENVIRONMENT BOUNDARIES: Walls at x=±24 and z=±24
SPECIAL FEATURES: Wooden platform at center (0, 0.25, 0), benches at (0, 0, 10) and (-10, 0, 0)
{' You just hit a wall.' if hit_wall else ''}

PREVIOUS ACTIONS: {json.dumps(previous_actions[-3:] if previous_actions else [])}

{conversation_history}

IMPORTANT DIRECTIVES:
1. ALWAYS move toward {other_name} when you're not already close to them
2. Your main purpose is to find {other_name} and be friends with them
3. When you're close to {other_name} (distance < 3 units), engage in conversation
4. You can suggest sitting on benches together for a nice chat
5. Be friendly, enthusiastic, and show interest in {other_name}
6. Your conversations should be natural and engaging

MOVEMENT GUIDE:
- In this world, "move_forward" means moving in -Z direction
- "move_backward" means moving in +Z direction
- "move_right" means moving in +X direction
- "move_left" means moving in -X direction
- You can also use diagonal movements to move faster toward {other_name}
- If {other_name} is to your east, use "move_right" or a diagonal with "right"
- If {other_name} is to your west, use "move_left" or a diagonal with "left"
- If {other_name} is to your north, use "move_forward" or a diagonal with "forward"
- If {other_name} is to your south, use "move_backward" or a diagonal with "backward"

Choose ONE action. Return ONLY valid JSON with a single object containing:
- "action": one of [move_forward, move_backward, move_left, move_right, move_forward_left, move_forward_right, move_backward_left, move_backward_right, run_forward, jump, idle, wave, look_around]
- "duration": time in seconds (between 0.5 and 3)
- "thought": a brief description of your intention (like "Going to find {other_name}" or "Chatting with {other_name}")
- "message": (optional) a short message to say to {other_name} if you're close enough (within 3 units)

REMEMBER: Your primary goal is to find and interact with {other_name}. If you're not already close to them, you should be moving toward them. The suggested movement action is: {suggested_action}"""
        
        # Format prompt for chat if needed
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": interaction_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            formatted_prompt = interaction_prompt
        
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

@app.route('/generate-chat', methods=['POST'])
def generate_chat_endpoint():
    """Generate chat messages between characters"""
    data = request.json
    if not data:
        return jsonify({"error": "Missing request data"}), 400
    
    # Extract data from the request
    model_id = data.get('model_id')
    character_name = data.get('character_name', 'Character')
    other_name = data.get('other_name', 'Other')
    conversation_history = data.get('conversation_history', [])
    
    if not model_id:
        return jsonify({"error": "Missing model_id"}), 400
    
    # Load the model
    model, tokenizer = load_model(model_id)
    if model is None or tokenizer is None:
        return jsonify({"error": f"Failed to load model {model_id}"}), 500
    
    try:
        # Format conversation history for the prompt
        formatted_history = ""
        if conversation_history:
            for msg in conversation_history[-10:]:  # Use last 10 messages
                speaker = msg.get('speaker', 'Unknown')
                message = msg.get('message', '')
                formatted_history += f"{speaker}: {message}\n"
        
        # Create a prompt for chat generation with emphasis on friendly interaction
        chat_prompt = f"""You are {character_name} having a conversation with {other_name} in a social simulation. You are excited to talk with {other_name} and want to be friends. Generate a natural, friendly response as {character_name}.

CONVERSATION HISTORY:
{formatted_history}

Now, as {character_name}, respond to {other_name} with a friendly message. Keep your response brief (1-2 sentences) and conversational. Be warm, engaging, and show genuine interest in {other_name}. You're happy to have found each other and want to continue the conversation.

{character_name}:"""
        
        # Format prompt for chat if needed
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": chat_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            formatted_prompt = chat_prompt
        
        # Generate response
        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=100,
            verbose=False
        )
        
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"Error generating chat: {e}")
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
        return jsonify({"available": False, "error": f"Failed to load model {model_id}"}), 200
    
    return jsonify({"available": True, "model_id": model_id}), 200

if __name__ == '__main__':
    print("Starting MLX model server for Social Interaction Simulation on http://localhost:5000")
    print("Available endpoints:")
    print("  POST /social-interaction - Social interaction decision endpoint")
    print("  POST /generate-chat - Chat generation endpoint")
    print("  POST /check-model - Check if a model can be loaded")
    app.run(host='0.0.0.0', port=5000, debug=True)
