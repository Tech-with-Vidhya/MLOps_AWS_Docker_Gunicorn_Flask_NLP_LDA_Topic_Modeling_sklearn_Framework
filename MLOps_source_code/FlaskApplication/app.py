import os
from pathlib import Path
# import preprocess
from dotenv import load_dotenv
from flask import Flask, jsonify, request
load_dotenv(Path(".env"))
import datetime
import os
from threading import Thread

if os.environ.get("ENV", "dev") == "prod":
    load_dotenv(Path(".env.prod"))
if os.environ.get("ENV", "dev") == "dev":
    load_dotenv(Path(".env.dev"))

from logging_module import logger
import engine
from s3_permission_checks import s3_permission_checks
app = Flask(__name__)


# Health check route for load balancer to find the health status of the flask application
@app.route("/health-status")
def get_health_status():
    logger.debug("Health check api version 5")
    resp = jsonify({"status": "I am alive version 5"})
    resp.status_code = 200
    return resp
    

@app.route("/s3-path", methods=['POST'])
def put_s3_file():
    logger.debug("s3-path api started")
    # code to verify the s3 permissions before starting the ML process
    status, response = s3_permission_checks()
    if not status:
        resp = jsonify({"error": response})
        resp.status_code = 500
        return resp
    filename = os.path.basename(request.json['unquoted_path'])
    path = "topic_modeling_output_folder/" + f"{filename}/"+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Running the ML Process in background using thread.
    thread = Thread(target=engine.convert_documents, args=(request.json['unquoted_path'],path), daemon=True)
    thread.start()
    resp = jsonify({"status": "started processing the file, please check s3 after a while", "outputS3Path": path})
    resp.status_code = 200
    return resp


if __name__ == "__main__":
    app.run(debug=True)
