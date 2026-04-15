from flask import Flask, request, jsonify, send_file
import subprocess
import os

app = Flask(__name__)

@app.route('/run-script', methods=['POST'])
def run_script():
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON payload"}), 400

    command = data.get("command")
    cwd = data.get("cwd", ".")  # Default to current directory if not specified
    env_vars = data.get("env", {})

    if not command or not isinstance(command, list):
        return jsonify({"error": "Invalid or missing 'command'. Must be a list."}), 400

    try:
        # Merge provided env with existing env
        env = {**os.environ, **env_vars}

        # Run the command
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True
        )

        return jsonify({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/download', methods=['GET'])
def download_file():
    filepath = request.args.get("path")
    if not filepath:
        return jsonify({"error": "Missing 'path' query parameter"}), 400

    if not os.path.isfile(filepath):
        return jsonify({"error": f"File '{filepath}' does not exist"}), 404

    try:
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)

