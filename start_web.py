import time
import webbrowser
import threading
from web_app import app

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:5001")

if __name__ == '__main__':
    print("=" * 60)
    print("CHESS AR VISION - SERVEUR WEB LOCAL ACTIF")
    print("Acces : http://127.0.0.1:5001")
    print("=" * 60)
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
