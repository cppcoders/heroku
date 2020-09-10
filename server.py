from flask import Flask, request, redirect, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import base64
from io import BytesIO


app = Flask(__name__)

app.secret_key = "counter=0"

# a route where we will display a welcome message via an HTML template

message = "Home"


@app.route("/")
def index():
    return render_template('index.html', message="Home")


app.config["TEMPLATES_AUTO_RELOAD"] = True
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
model = load_model("static/model.hdf5")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    files = request.files.getlist('uploadFile')
    labels = {0: 'desert', 1: 'plant', 2: 'water'}
    pred = {}
    # if there is more than 10 uploaded images don't continue and return with this message ,.... مش وكالة من غير بواب هي
    if len(files) > 10:
        return redirect('/')

    for file in files:
        if not file.filename.endswith(ALLOWED_EXTENSIONS):
            continue
        img = Image.open(file)
        img = img.resize((256, 256))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        pim = base64.b64encode(buffered.getvalue())
        img = np.asarray(img, dtype=np.float32)
        img = img / 255
        img = img[..., :3]
        img = img.reshape(-1, 256, 256, 3)
        predict = model.predict(img)
        predict = labels[np.argmax(predict)]
        pred[pim] = predict
    return render_template('index.html', message=pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
