"""
Flask application for NBA MVP Predictor REST API.
Modular backend entry point for scalable deployment.
"""

import os
from flask import Flask

from api.routes import bp as api_bp


def create_app():
    """Application factory for modular backend architecture."""
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix="/api")
    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
