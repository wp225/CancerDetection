import base64
import io
import logging
import os
from cnnClassifier import logger
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

from cnnClassifier.pipeline.prediction import PredictionPipeline

UPLOAD_FOLDER = './upload'

app = FastAPI()

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Serve the HTML form for uploading images
@app.get("/", response_class=HTMLResponse)
async def serve_form():
    return """
    <!DOCTYPE html>
<html>
<head>
    <title>Image Upload</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            color: #333;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0;
            padding: 20px;
        }
        h1, h2 {
            color: #4CAF50;
        }
        form {
            margin-bottom: 20px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        input[type="file"] {
            padding: 5px;
            margin-top: 10px;
            margin-bottom: 20px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .image-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
            width: 100%;
        }
        .image-container div {
            text-align: center;
            flex: 1;
        }
        .image-container img {
            width: 400px; /* Set the width */
            height: 400px; /* Set the height */
            object-fit: cover; /* Ensure the image covers the entire area */
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        #prediction-result {
            font-weight: bold;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Upload an Image</h1>
    <form id="upload-form" action="/upload_image/" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <button type="submit">Upload</button>
    </form>
    <h2>Prediction Result</h2>
    <p id="prediction-result"></p>
    <div class="image-container">
        <div>
            <h3>Uploaded Image</h3>
            <img id="uploaded-image" src="" alt="Uploaded Image" style="display:none;">
        </div>
        <div>
            <h3>Grad-CAM Image</h3>
            <img id="grad-cam-image" src="" alt="Grad-CAM Image" style="display:none;">
        </div>
    </div>
    <script>
        document.getElementById('upload-form').onsubmit = async function(event) {
            event.preventDefault();
            const formData = new FormData(this);
            const response = await fetch('/upload_image/', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            document.getElementById('prediction-result').innerText = result.info;

            const uploadedImage = document.getElementById('uploaded-image');
            uploadedImage.src = URL.createObjectURL(formData.get('file'));
            uploadedImage.style.display = 'block';

            if (result.grad_cam_image) {
                const gradCamImage = document.getElementById('grad-cam-image');
                gradCamImage.src = 'data:image/png;base64,' + result.grad_cam_image;
                gradCamImage.style.display = 'block';
            }
        }
    </script>
</body>
</html>

"""


@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type.startswith("image/"):
        try:
            file_location = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_location, "wb") as f:
                f.write(await file.read())
            logger.info(file_location)
            print(file_location)
            classifier = PredictionPipeline(file_location)
            logger.info(classifier)
            try:
                pred, grad_cam_image = classifier._predict()
                print(pred)
                logging.info(pred)
            except Exception as e:
                logger.info('failed to predict')
                logger.info(e)

            buffered = io.BytesIO()
            grad_cam_image.save(buffered, format="PNG")
            grad_cam_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return {
                "info": pred,
                "filename": file.filename,
                "grad_cam_image": grad_cam_image_base64,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File save or prediction error: {e}")
    else:
        raise HTTPException(status_code=400, detail="File must be an image")


@app.get("/train")
def trainRoute():
    # os.system("python main.py")
    os.system("dvc repro")
    return "Training done successfully!"
