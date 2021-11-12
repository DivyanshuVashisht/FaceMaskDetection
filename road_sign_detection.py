from keras.models import model_from_json
import numpy as np
import cv2

json_file = open("./model.json")
loaded_model_json = json_file.read()
json_file.close();
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./weights.h5")
print("Loaded Model Successfully")

def getClassName(classNo):
    if classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo ==1: return 'Speed Limit 30 km/h'
    elif classNo ==2: return 'Speed Limit 50 km/h'
    elif classNo ==3: return 'Speed Limit 60 km/h'
    elif classNo ==4: return 'Speed Limit 70 km/h'
    elif classNo ==5: return 'Speed Limit 80 km/h'
    elif classNo ==6: return 'End of Speed Limit 80 km/h'
    elif classNo ==7: return 'Speed Limit 100 km/h'
    elif classNo ==8: return 'Speed Limit 120 km/h'
    elif classNo ==9: return 'No passing'
    elif classNo ==10: return 'No passing for vehicles over 3.5 metric tons'
    elif classNo ==11: return 'Right-of-way at the next intersection'
    elif classNo ==12: return 'Priority road'
    elif classNo ==13: return 'Yield'
    elif classNo ==14: return 'Stop'
    elif classNo ==15: return 'No vehicles'
    elif classNo ==16: return 'Vehicles over 3.5 metric tons prohibited'
    elif classNo ==17: return 'No entry'
    elif classNo ==18: return 'General caution'
    elif classNo ==19: return 'Dangerous curve to the left'
    elif classNo ==20: return 'Dangerous curve to the right'
    elif classNo ==21: return 'Double curve'
    elif classNo ==22: return 'Bumpy curve'
    elif classNo ==23: return 'Slippery road'
    elif classNo ==24: return 'Road narrows on the right'
    elif classNo ==25: return 'Road work'
    elif classNo ==26: return 'Traffic signals'
    elif classNo ==27: return 'Pedestrians'
    elif classNo ==28: return 'Children crossing'
    elif classNo ==29: return 'Bicycles crossing'
    elif classNo ==30: return 'Beware of ice/snow'
    elif classNo ==31: return 'Wild animals crossing'
    elif classNo ==32: return 'End of all speed and passing limits'
    elif classNo ==33: return 'Turn right ahead'
    elif classNo ==34: return 'Turn left ahead'
    elif classNo ==35: return 'Ahead only'
    elif classNo ==36: return 'Go straight or right'
    elif classNo ==37: return 'Go straight or left'
    elif classNo ==38: return 'Keep right'
    elif classNo ==39: return 'Keep left'
    elif classNo ==40: return 'Roundabout mandatory'
    elif classNo ==41: return 'End of no passing'
    elif classNo ==42: return 'End of no passing by vehicles over 3.5 metric tons'

def preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#converting the image to grayscale
    image = image/255 #normalization
    return image 

capt = cv2.VideoCapture(0)
#0 - capture the video from webcam
capt.set(3, 640)
capt.set(4, 480)
capt.set(10, 180)

while True:
    #capture a single image frame from the video
    message, image = capt.read()
    #convert image to array
    imagearr = np.asarray(image)
    #resizing the image to original size
    imagearr = cv2.resize(int(imagearr), (32, 32))
    #converting the image to grayscale
    imagearr = preprocessing(imagearr)
    #change the dimensions to the original data
    imagearr = imagearr.reshape(1, 32, 32, 1)
    #pass the image to prediction model
    predictions = loaded_model.predict(imagearr)
    #get the index of the value which has highest probability
    classIndex = loaded_model.predict_classes(imagearr)
    #put a text in the original image(class, probability, class_value, %accuracy)
    cv2.putText(image, "Class: ", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "Probability: ", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    probabilityValue = np.amax(predictions)
    
    if probabilityValue > 0.75:
        cv2.putText(image, getClassName(classIndex), (120, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, str(int(probabilityValue)*100)+"%", (200, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Model Predictions", image)
    cv2.waitKey(1)
    if returnedValue == ord("q") or returnedValue == ord("Q"):
        cv2.destroyAllWindows()
        break