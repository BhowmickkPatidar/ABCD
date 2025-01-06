from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from groq import Groq
import markdown2

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        message = request.json.get("message")
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        # Debugging information
        print(f"Received message: {message}")
        
        # Create chat completion using Groq client
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a highly skilled CA Drafting AI Assistant. "
                        "Your primary role is to assist users in drafting accurate, professional, and well-structured documents. "
                        "Always use headings, bullet points, numbered lists, and proper formatting. "
                        "For agreements, follow templates with clear sections, subclauses, and formal language."
                    )
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=1000
        )
        
        response = chat_completion.choices[0].message.content
        
        # Convert Markdown to HTML
        response_html = markdown2.markdown(response)
        
        return jsonify({"response": response_html})
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    api_key_status = bool(os.environ.get("GROQ_API_KEY"))
    return jsonify({
        "status": "healthy",
        "api_key_configured": api_key_status,
        "model": "llama-3.3-70b-versatile"
    }), 200



# In production, Gunicorn will use `create_app()` to run the application
if __name__ == "__main__":
    # This is for local development, use Gunicorn in production
    app.run()
