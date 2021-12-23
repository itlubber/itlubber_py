import os
import json
from flask_cors import CORS
from flask import Flask, request


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/api/test_handler", methods=["GET", "POST"], strict_slashes="false")
def test_handler():
    if request.method == "GET":
        content = request.args.get("content")

    if request.method == "POST":
        content = request.get_json().get("content")

    res = {"content": content, "result": "this is a test api."}
    return json.dumps({"code": 200, "msg": "success", "data": res}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, threaded=True, debug=False, use_reloader=False)
