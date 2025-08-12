import os
from flask import Flask, request, send_from_directory, jsonify
import csv

app = Flask(__name__, static_folder='UI')

# Absolute path to the schedules directory
SCHEDULES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../schedules/participant_schedules'))

# Serve the main automated experiment UI
@app.route('/')
def index():
    return send_from_directory('UI', 'AutomatedExperiment.html')

# Serve static files (JS, CSS, images)
@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('UI', path)

# Serve schedules
@app.route('/schedules/<path:filename>')
def serve_schedule(filename):
    print("Requested schedule file:", filename)
    print("Serving from:", SCHEDULES_DIR)
    return send_from_directory(SCHEDULES_DIR, filename)

# Receive gesture logs from the UI
@app.route('/log_gesture', methods=['POST'])
def log_gesture():
    data = request.get_json()
    pid = data.get('pid')
    gesture = data.get('gesture')
    gesture_index = data.get('gesture_index')
    timestamp = data.get('timestamp')
    print(f"Received log: PID={pid}, Gesture={gesture}, Index={gesture_index}, Time={timestamp}")
    # Write to CSV
    os.makedirs('logs', exist_ok=True)
    log_path = f'logs/auto_labels_{pid}.csv'
    write_header = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['Timestamp', 'Gesture', 'Gesture_Index', 'Participant_ID'])
        writer.writerow([timestamp, gesture, gesture_index, pid])
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True) 