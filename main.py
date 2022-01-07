from flask import Flask,jsonify,request
import flask
from classifier import get_Prediction
#__name__ is constructor for flask
app = Flask(__name__)
#Route for app which is predict digit by post method
@app.route('/predict-digit',methods=['POST'])
def prefict_data():
    #Getting digit files 
    image = request.files.get('alphabet')
    #image is parameter for prediction
    prediction = get_Prediction(image)
    #return prediction value and 200 is for success msg (status code)
    return jsonify({
        'prediction' : prediction
    }),200
#run the app
if __name__=='__main__':
    #Condition if name is main
    app.run(debug=True)