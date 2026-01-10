from flask import Blueprint, request, jsonify, render_template
from app.services.scraper_service import scrape_data
from app.services.analysis_service import analysis_anime, fetch_analysis
from app.services.filter_service import filter_anime
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

@routes.route("/analysis", methods=['POST'])
def analysis_endpoint():
    response = request.get_json()

    result = analysis_anime(response)

    return jsonify(json.loads(json.dumps(result, indent=4, sort_keys=False)))

@routes.route("/filter", methods=['POST'])
def filter_endpoint():
    response = request.get_json()
    selected_genres = response.get("selected_genres", [])
    selected_studios = response.get("selected_studios", [])
    data = response.get("data", [])

    result = filter_anime(data, selected_genres, selected_studios)

    return jsonify(json.loads(json.dumps(result, indent=4, sort_keys=False)))

@routes.route("/recommendation", methods=['POST'])
def recommend_endpoint():
    response = request.get_json()

    result = fetch_analysis(response)

    return jsonify(json.loads(json.dumps(result, indent=4, sort_keys=False)))