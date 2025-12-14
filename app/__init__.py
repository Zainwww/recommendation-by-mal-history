from flask import Flask
from dotenv import load_dotenv
import os

def create_app():
    load_dotenv()

    app = Flask(__name__)

    # register blueprint
    from app.routes.main_route import routes
    app.register_blueprint(routes)

    return app
