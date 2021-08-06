import cv2
import numpy as np
import tensorflow as tf
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#load model
def initialize_model():
    model = tf.keras.models.load_model('myModel.h5')
    return model




# preprocessing

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 1)  # gaussian blur
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 9, 2) # apply adaptive threshold
    return img_threshold

## finding the biggest contour
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area>100:
            peri = cv2.arcLength(i,True)
            #find the corners it will approximate the poly count
            approx = cv2.approxPolyDP(i, 0.02*peri,  True) #resolution is 0.02*arclength
            print(len(approx))
            # if len(approx == 4):
            #     if area>max_area:
            if area >max_area and len(approx == 4): #if poly count is 4 its either square or rectangle
                    biggest = approx
                    max_area = area
            # else:
            #     biggest = 0
            #     max_area = 0
    return biggest,max_area

def reorder(mypoints):
    mypoints = mypoints.reshape((4, 2))
    mypointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = mypoints.sum(1)
    diff = np.diff(mypoints, axis=1)
    mypointsNew[0] = mypoints[np.argmin(add)]
    mypointsNew[3] = mypoints[np.argmax(add)]
    mypointsNew[1] = mypoints[np.argmin(diff)]
    mypointsNew[2] = mypoints[np.argmax(diff)]
    return mypointsNew

def splitBoxes(img):
    rows = np.vsplit(img, 9)
    #print(rows)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes

def get_prediction (boxes,model):
    result = []
    for image in boxes:
        ## prepare image
        img = np.asarray(image)
        img = img[4:img.shape[0]-4, 4:img.shape[1]-4]
        img = cv2.resize(img,(28,28))
        img = img/255
        img = img.reshape(1,28,28,1)
        ###
        pred = model.predict(img)
        classindex = np.argmax(pred, axis=-1)
        prob = np.amax(pred)
        print(classindex, prob)

        if prob > 0.8:
            result.append(classindex[0])
        else:
            result.append(0)
    return  result


#display number on image
def displayNumber(img, numbers, color=(0,255,0)):
    #each small boxes shape is (50,50)
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    # print(img.shape[0], img.shape[1])
    # print('hi',secH,secW)
    for x in range(0,9):
        for y in range(0,9):
            if numbers[(y*9)+x]!= 0:
                cv2.putText(img, str(numbers[(y*9)+x]),
                            (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


#def drawGrid():









