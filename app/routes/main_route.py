from flask import Blueprint, request, jsonify, render_template
from app.services.scraper_service import scrape_data
import json

routes = Blueprint("main", __name__)

@routes.route('/')
def main():
    return render_template("index.html")

@routes.route("/scrape", methods=["GET"])
def scrape_endpoint():
    username = request.args.get("username")

    if not username:
        return jsonify({"error": "Please provide a username via ?username="}), 400

    result = scrape_data(username)

    return jsonify(json.loads(json.dumps(result, indent=4, sort_keys=False)))
