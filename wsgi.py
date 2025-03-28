import os
from app import app

# Expose the Flask app as 'application' for Render
application = app

if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    # Get host from environment variable or default to 0.0.0.0
    host = os.environ.get("HOST", "0.0.0.0")
    
    application.run(host=host, port=port) 