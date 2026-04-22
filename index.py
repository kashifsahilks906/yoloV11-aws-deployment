from flask import Flask, request, jsonify, send_file, make_response, render_template
from flask_cors import CORS
import io
import json
from inference import predict

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        annotated_image, detection_results = predict(image)

        image_buffer = io.BytesIO(annotated_image)
        image_buffer.seek(0)

        response = make_response(
            send_file(image_buffer, mimetype='image/jpeg')
        )

        response.headers['Detection-Results'] = json.dumps(detection_results)

        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)