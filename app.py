from flask import Flask, request, render_template, jsonify
from mlp import get_recommender
from difflib import get_close_matches

app = Flask(__name__)

# builds the recommender once at startup
recommender = get_recommender()


@app.route('/', methods=['GET'])
def index():
  return render_template('index.html', titles=recommender.all_titles, base_title=None, recs=None)

@app.get("/api/suggest")
def api_suggest():
    q = (request.args.get("q") or "").strip().lower()
    if not q:
        return jsonify({"suggestions": []})
    # contains-matching
    matches = [t for t in recommender.all_titles if q in t.lower()]
    return jsonify({"suggestions": matches[:20]})

@app.route('/recommend', methods=['POST'])
def recommend_route():
  query = (request.form.get('movie') or '').strip()
  title = None

  # exact match (case-insensitive)
  if query:
    lowered = query.lower()
    title = next((t for t in recommender.all_titles if t.lower() == lowered), None)

  # substring contains
  if not title and query:
    candidates = [t for t in recommender.all_titles if lowered in t.lower()]
    if candidates:
      title = candidates[0]

  # fuzzy match to the closest title
  if not title and query:
    guess = get_close_matches(query, recommender.all_titles, n=1, cutoff=0.5)
    title = guess[0] if guess else None

  if title:
    recs = recommender.recommend_titles(title, top_k=5)
    base = title
  else:
    recs, base = [], None
  return render_template('index.html', titles=recommender.all_titles, base_title=base, recs=recs)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)