import os
import logging
from flask import Flask
from flask_socketio import SocketIO

# Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create Flask App
app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['SECRET_KEY'] = 'secret!'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['JSON_AS_ASCII'] = False  # Critical for Meitei Mayek rendering

# Initialize SocketIO
# max_http_buffer_size increased for base64 JPEG frame transport (CNN+BiLSTM pipeline)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                    max_http_buffer_size=5 * 1024 * 1024)  # 5 MB

# 1. Load Application Core
from src.core import load_models

# 2. Import Blueprints
from src.routes.pages import pages_bp
from src.routes.api import api_bp
from src.routes.sockets import register_socket_handlers

# 3. Register Routes & Handlers
app.register_blueprint(pages_bp)
app.register_blueprint(api_bp)
register_socket_handlers(socketio)

# 4. Load models at import time so all launchers (run.py, app.py) get them
load_models()

if __name__ == '__main__':
    # Note: the actual server execution happens via run.py at the root
    socketio.run(app, debug=True, use_reloader=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
