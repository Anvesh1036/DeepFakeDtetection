from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from predict import predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction = predict_image(filepath)
            return render_template('result.html', prediction=prediction, image_url=filepath)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
