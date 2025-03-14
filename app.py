from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
import requests
import json
import os
from rapidfuzz import fuzz

# Initialize Flask
app = Flask(__name__)

# Load API Keys from Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"

# Initialize Firebase (Ensure correct credentials loading)
firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
if firebase_credentials:
    cred = credentials.Certificate(json.loads(firebase_credentials))
    firebase_admin.initialize_app(cred)
else:
    raise ValueError("Firebase credentials not found!")

db = firestore.client()

# Configure Google Gemini API
genai.configure(api_key=GEMINI_API_KEY)


def search_firebase(location, problem, threshold=70):
    """Search for complaints in Firebase that match the given location and problem."""
    location = location.strip().title()
    complaints_ref = db.collection("complaints")
    
    query = complaints_ref.where("location", ">=", location).where("location", "<=", location + "\uf8ff")
    results = list(query.stream())

    matched_complaints = []
    for complaint in results:
        data = complaint.to_dict()
        similarity_score = fuzz.token_sort_ratio(problem.lower(), data["text"].lower())
        if similarity_score >= threshold:
            matched_complaints.append({**data, "similarity": similarity_score})

    matched_complaints.sort(key=lambda x: x["similarity"], reverse=True)
    return matched_complaints


def search_online(location, problem):
    """Search online for news related to the given problem and location."""
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    query = f"{problem} in {location}"
    data = {"q": query, "num": 5}

    response = requests.post(SERPER_URL, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        results = response.json().get("organic", [])
        return [
            {"title": r.get("title", "No Title"), "link": r.get("link", "#"), "snippet": r.get("snippet", "No snippet available.")}
            for r in results
        ]
    
    return []


def generate_summary(location, problem, complaints, news_results):
    """Generate a summary using Google Gemini AI."""
    news_text = "\n".join(
        [f"- **{news['title']}**: {news['snippet']} ([Source]({news['link']}))" for news in news_results]
    )

    prompt = f"""
    You are an expert in analyzing social issues. A complaint about "{problem}" was received in {location}.
    Below is relevant information:

    - **Database complaints**: {complaints if complaints else 'None found'}
    - **Relevant News Articles**:
      {news_text if news_text else 'No relevant news found'}

    Please provide:
    1. Possible reasons for this issue.
    2. Suggested solutions.
    3. Additional recommendations.
    """

    try:
        model = genai.GenerativeModel("gemini-pro-latest")  # Fixed model name
        response = model.generate_content(prompt)
        return response.text if response else "No summary available."
    except Exception as e:
        return f"Error generating summary: {str(e)}"


@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze complaints and provide recommendations."""
    try:
        data = request.get_json()
        location = data.get("location", "").strip().title()
        problem = data.get("problem", "").strip()

        if not location or not problem:
            return jsonify({"error": "Both 'location' and 'problem' fields are required!"}), 400

        complaints = search_firebase(location, problem)
        news_results = search_online(location, problem)
        summary = generate_summary(location, problem, complaints, news_results)

        return jsonify({
            "location": location,
            "problem": problem,
            "matched_complaints": complaints,
            "news_results": news_results,
            "summary": summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    """Home route"""
    return jsonify({"message": "Welcome to the Complaint Analysis API!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
