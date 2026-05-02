from flask import Blueprint, render_template, Response
from src.core import generate_frames

pages_bp = Blueprint('pages', __name__)

@pages_bp.route('/')
def index():
    return render_template('landing.html')

@pages_bp.route('/app')
def application():
    return render_template('app.html')

@pages_bp.route('/video_feed')
def video_feed():
    """Video streaming route for the legacy server-side processing."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
