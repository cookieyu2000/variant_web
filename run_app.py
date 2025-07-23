# pubtator_inference/run_app.py

from pubtator.app import app

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)

# -------Linux command to run the app-------
# gunicorn -w 4 -b 0.0.0.0:8080 run_app:app
# -------Windows command to run the app-------
# waitress-serve --listen=10.22.24.176:8080 run_app:app
