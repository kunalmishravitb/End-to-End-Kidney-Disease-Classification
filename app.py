from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline






os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')



app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg" # input image
        self.classifier = PredictionPipeline(self.filename) # classifier object


# default route
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')



# train route
@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    #os.system("python main.py")
    os.system("dvc repro")
    return "Training done successfully!"



# predict route
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image'] # get image from user
    decodeImage(image, clApp.filename) # decode a base64 encoded image and save it to a file
    result = clApp.classifier.predict() # predict the image
    return jsonify(result) # return the result



# port
if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080, debug=True)

