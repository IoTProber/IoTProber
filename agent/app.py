"""
Flask Backend Server for IoT Query Decomposition System
Provides API endpoints for query decomposition using the DecompositionAgent
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json
from datetime import datetime

# Add the current directory to the path to import decomposition module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from decomposition import DecompositionAgent

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the DecompositionAgent
# You can change the LLM model here: "gemini", "deepseek", or "openai"
agent = None
current_llm = "deepseek"  # Default LLM

def initialize_agent(llm_type="deepseek"):
    """Initialize or reinitialize the agent with specified LLM"""
    global agent, current_llm
    try:
        print(f"Initializing agent with {llm_type} model...")
        agent = DecompositionAgent(llm=llm_type)
        current_llm = llm_type
        print(f"Agent initialized successfully with {llm_type}")
        return True
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return False

# Initialize agent on startup
initialize_agent(current_llm)

# ==================== API Routes ====================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('web', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS)"""
    return send_from_directory('web', path)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'agent_initialized': agent is not None,
        'current_llm': current_llm,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/decompose', methods=['POST'])
def decompose_query():
    """
    Main endpoint for query decomposition
    
    Request body:
    {
        "query": "user query string",
        "rule_matched": false (optional),
        "langchain_version": "1.2" (optional)
    }
    
    Response:
    {
        "success": true/false,
        "result": {...} or null,
        "error": "error message" (if failed)
    }
    """
    try:
        # Check if agent is initialized
        if agent is None:
            return jsonify({
                'success': False,
                'error': 'Agent not initialized. Please check server logs.'
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: query'
            }), 400
        
        query = data['query'].strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        # Get optional parameters
        rule_matched = data.get('rule_matched', False)
        langchain_version = data.get('langchain_version', '1.2')
        
        # Perform decomposition
        print(f"\n{'='*50}")
        print(f"Processing query: {query}")
        print(f"Rule matched: {rule_matched}")
        print(f"LangChain version: {langchain_version}")
        print(f"{'='*50}\n")
        
        # Call the decomposition agent
        if current_llm == "deepseek":
            result = agent.decompose_query(
                query, 
                rule_matched=rule_matched, 
                langchain_version="0.3.27"
            )
        else:
            result = agent.decompose_query(
                query, 
                rule_matched=rule_matched,
                langchain_version=langchain_version
            )
        
        print(f"\nDecomposition result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n{'='*50}\n")
        
        # Save the result
        agent.save_history()
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in decompose_query: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/retrieve', methods=['POST'])
def retrieve_query():
    """
    Main endpoint for query decomposition and retrieval
    
    Request body:
    {
        "query": "user query string",
        "fingerprint": {
            "ip": "192.168.1.1",
            "port": 8080,
            ...
        }
    }
    
    Response:
    {
        "success": true/false,
        "result": {
            "decomposition_result": {...},
            "retrieval_result": {
                "local_result": [...],
                "community_result": [...],
                "reasoning_result": {...}
            },
            "from_cache": true/false
        },
        "error": "error message" (if failed)
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'query' not in data or 'fingerprint' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: query and fingerprint'
            }), 400
        
        query = data['query'].strip()
        fingerprint = data['fingerprint']
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        if not isinstance(fingerprint, dict):
            return jsonify({
                'success': False,
                'error': 'Fingerprint must be a JSON object'
            }), 400
        
        if 'ip' not in fingerprint:
            return jsonify({
                'success': False,
                'error': 'Fingerprint must contain ip field'
            }), 400
        
        # Perform retrieval
        print(f"\n{'='*60}")
        print(f"Processing retrieval query")
        print(f"Query: {query}")
        print(f"Fingerprint: {json.dumps(fingerprint, indent=2, ensure_ascii=False)}")
        print(f"{'='*60}\n")
        
        # Import and call retrieval.main
        from retrieval import main as retrieval_main
        
        # Call retrieval main function with query and fingerprint
        result = retrieval_main([query], query_fingerprint=fingerprint)
        
        print(f"\nRetrieval result:")
        print(f"Has decomposition result: {result.get('decomposition_result') is not None}")
        print(f"Has retrieval result: {result.get('retrieval_result') is not None}")
        print(f"From cache: {result.get('from_cache', False)}")
        print(f"\n{'='*60}\n")
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in retrieve_query: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """
    Get decomposition history
    
    Response:
    {
        "success": true,
        "history": [...]
    }
    """
    try:
        if agent is None:
            return jsonify({
                'success': False,
                'error': 'Agent not initialized'
            }), 500
        
        history = agent.get_decomposition_history()
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        print(f"Error in get_history: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """
    Clear decomposition history
    
    Response:
    {
        "success": true,
        "message": "History cleared"
    }
    """
    try:
        if agent is None:
            return jsonify({
                'success': False,
                'error': 'Agent not initialized'
            }), 500
        
        agent.clear_history()
        
        return jsonify({
            'success': True,
            'message': 'History cleared successfully'
        })
        
    except Exception as e:
        print(f"Error in clear_history: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/known-problems', methods=['GET'])
def get_known_problems():
    """
    Get all known problem types
    
    Response:
    {
        "success": true,
        "problems": {...}
    }
    """
    try:
        if agent is None:
            return jsonify({
                'success': False,
                'error': 'Agent not initialized'
            }), 500
        
        problems = agent.get_known_problems()
        
        return jsonify({
            'success': True,
            'problems': problems,
            'count': len(problems)
        })
        
    except Exception as e:
        print(f"Error in get_known_problems: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """
    Get current configuration
    
    Response:
    {
        "success": true,
        "config": {...}
    }
    """
    return jsonify({
        'success': True,
        'config': {
            'current_llm': current_llm,
            'available_llms': ['gemini', 'deepseek', 'openai'],
            'agent_initialized': agent is not None
        }
    })

@app.route('/api/config/llm', methods=['POST'])
def change_llm():
    """
    Change the LLM model
    
    Request body:
    {
        "llm": "gemini" | "deepseek" | "openai"
    }
    
    Response:
    {
        "success": true/false,
        "message": "..."
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'llm' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: llm'
            }), 400
        
        llm_type = data['llm']
        
        if llm_type not in ['gemini', 'deepseek', 'openai']:
            return jsonify({
                'success': False,
                'error': 'Invalid LLM type. Must be one of: gemini, deepseek, openai'
            }), 400
        
        # Reinitialize agent with new LLM
        success = initialize_agent(llm_type)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'LLM changed to {llm_type} successfully',
                'current_llm': current_llm
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to initialize agent with new LLM'
            }), 500
        
    except Exception as e:
        print(f"Error in change_llm: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ==================== Main ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("IoT Query Decomposition System - Backend Server")
    print("="*60)
    print(f"Current LLM: {current_llm}")
    print(f"Agent initialized: {agent is not None}")
    print("\nStarting server on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader to prevent double initialization
    )
