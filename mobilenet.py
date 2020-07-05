import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils 
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS
import tensorflow as tf
# from keras import backend as K

app = Flask(__name__)
CORS(app)

def get_model():
    global model
    model = load_model('food_model_173.h5')
    print(" * Model loaded!")

print(" * Loading Keras model...")
get_model()

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('uint8')
    image = imagenet_utils.preprocess_input(image)
    print("img to array",image)
    return image
    

@app.route("/mobilenet", methods=["POST"])
def predict():
        # data = {"success": False}
    #if Flask.request.method == "POST":
        message = request.get_json(force=True)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))

        processed_image = preprocess_image(image, target_size=(224, 224))

        prediction = model.predict(processed_image)
        print("**type of array:",type(prediction))
        print("**prediction:",prediction)
        index = np.argmax(prediction,axis=-1)
        print("*******Predicted index:",index)
        food_list=['aalo sabzi','bajri ka rotla','butter chicken','chapati','chinese roll','chole bhature','dabeli','dahi vada','dal bati','dhokla','dosa','fried rice','gajar halwa','gulab jamun','hot and sour soup','idli','jalebi','kadhi','kaju katri','moti chur ka ladoo','omelette','paneer ka sabzi','pani puri','pav bhaji','poha','samosa','tea','thepla','upma','vada']
        
        calorie={'aalo sabzi':99,'bajri ka rotla':138,'butter chicken':202,'chapati':104,'chinese roll':109,'chole bhature':497,'dabeli':250,'dahi vada':76,'dal bati':340,'dhokla':160,'dosa':133,'fried rice':168,'gajar halwa':175,'gulab jamun':149,'hot and sour soup':60,'idli':33,'jalebi':300,'kadhi':151,'kaju katri':504,'moti chur ka ladoo':122,'omelette':154,'paneer ka sabzi':289,'pani puri':36,'pav bhaji':107,'poha':250,'samosa':262,'tea':3,'thepla':120,'upma':209,'vada':73}
        
        print("************index:",index[0])
        pred_value = food_list[index[0]]
        print("@@@@@@@@@@Predicted class@@@@@@@@@@@",pred_value)
        calc=calorie.get(pred_value)
        print("***********calorie:",calc)
        #results = imagenet_utils.decode_predictions(prediction)
        #data["predictions"] = []

        #for (imagenetID, label, prob) in results[0]:
                # r = {"label": label, "probability": float(prob)}
                # data["predictions"].append(r)

        # data["success"] = True            
    
        response = {
                 'prediction': {
                    'pred': pred_value,
                    'calorie':calc
            
                 }
        }
        return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)

