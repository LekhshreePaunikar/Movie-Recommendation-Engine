from flask import Flask, request, render_template, jsonify
from mlp import get_recommender
from difflib import get_close_matches # available if we later want fuzzy suggestions

app = Flask(__name__)

# builds the recommender once at startup
recommender = get_recommender()

# Serve the homepage with the dropdown of all titles
@app.route('/', methods=['GET'])
def index():
  return render_template('index.html', titles=recommender.all_titles, base_title=None, recs=None)

# Lightweight live suggestions: simple contains-match, case-insensitive
@app.get("/api/suggest")
def api_suggest():
    q = (request.args.get("q") or "").strip().lower()
    if not q:
        return jsonify({"suggestions": []})
    # contains-matching, case-insensitive
    matches = [t for t in recommender.all_titles if q in t.lower()]
    return jsonify({"suggestions": matches[:20]})

# Handle the recommend form: find exact title, then ask the model for top-5 similar
@app.route('/recommend', methods=['POST'])
def recommend_route():

  query = (request.form.get('movie') or '').strip()
  if not query:
      return render_template('index.html', base_title=None, recs=[], error_msg='Please type a movie name.')

  # strict exact and case-insensitive match only; no auto-correction
  lowered = query.lower()
  title = next((t for t in recommender.all_titles if t.lower() == lowered), None)

  if title:
    recs = recommender.recommend_titles(title, top_k=5)
    return render_template('index.html', base_title=title, recs=recs, error_msg=None)
  else:
      return render_template('index.html', base_title=None, recs=[], error_msg=f'no movie of name "{query}" can be found!')

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)