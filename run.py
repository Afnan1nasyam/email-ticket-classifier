"""Development entrypoint.

Runs the Flask development server (Windows-friendly). Production uses gunicorn
via the Dockerfile, not this script.
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    # debug=False so failures return JSON errors, never the interactive debugger.
    app.run(host="127.0.0.1", port=5000, debug=False)
