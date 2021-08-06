from flask import  Flask,render_template, url_for, request, redirect, make_response
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from static.upload_folder.utils import *
from static.upload_folder.sudoku_solver import *
import cv2

app =  Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']= -1
#app.secret_key = "secret key"
print('modelloaded')
upload_folder = 'deployment/static/upload_folder'
app.config['upload'] = upload_folder
model_path = 'D:/opencv/model.h5'





def predict():
    # path = '/depoyment/static'
    # image = os.path.join(path,'image.jpg')
    # array = np.asarray(image)
    # print(array.shape)
    ori = cv2.imread('./static/image.jpg')
    image = cv2.imread('./static/image.jpg',0)
    image = cv2.resize(image, (28,28))
    print(image.shape)
    image = np.reshape(image,(1,28,28,1))
    model = load_model('myModel.h5')
    predictions = model.predict(image)
    predict = np.argmax(predictions,axis=-1)
    return predict
    #return render_template('predict.html', display = ori, predict = predict)


def predict_sudoku():

    # image_path = 'sudoku/Resources/first.jpg'\
    ori = cv2.imread('./static/sudoku.jpg')
    height_img = 450
    width_img = 450
    model = initialize_model()
    # print(model.summary())
    #
    # video_stream = cv2.VideoCapture(0)
    # while True:
    #     success, imag = video_stream.read()
    #     cv2.imshow('pred', imag)
    # # prepare image
    #img = cv2.imread(ori)
    print(ori.shape)
    img = cv2.resize(ori, (width_img, height_img))  # resize the image
    img_blank = np.zeros((height_img, width_img, 3), np.uint8)  # blank image to test debug
    img_threshold = preprocess(img)
    cv2.imshow('1', img_threshold)
    #
    #     #find contours
    #
    imgContours = img.copy()
    imgBigcontour = img.copy()
    contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)
    cv2.imshow('2', imgContours)
    #
    #     #find the biggest contour
    biggest, area = biggestContour(contours)
    print(biggest, area)
    print(biggest.shape)
    # cv2.imshow('3', biggest)
    if biggest.size != 0:
        biggest = reorder(biggest)
        print(biggest)
        print(biggest.ndim)

        cv2.drawContours(imgBigcontour, biggest, -1, (0, 255, 0), 10)
        cv2.imshow('3', imgBigcontour)
        pts1 = np.float32(biggest)  # points for warp
        pts2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgwarpcolored = cv2.warpPerspective(img, matrix, (width_img, height_img))
        imgdetecteddigits = img_blank.copy()
        imgwarpcolored = cv2.cvtColor(imgwarpcolored, cv2.COLOR_BGR2GRAY)

        #
        #         ##small boxes and find digit
        boxes = splitBoxes(imgwarpcolored)
        numbers = get_prediction(boxes, model)
        print(len(boxes))
        print(np.asarray(boxes[1]).shape)
        print(numbers)
        imgdetecteddigits = displayNumber(imgdetecteddigits, numbers, color=(255, 0, 255))
        numbers = np.asarray(numbers)
        pos = np.where(numbers > 0, 0, 1)
        #
        #
        solvedimgfinal = imgwarpcolored.copy()
        #
        #         ##split numbers as array
        board = np.array_split(numbers, 9)
        try:
            solve(board)
        except:
            pass
        print(board)

        flat = []
        for x in board:
            for y in x:
                flat.append(y)
        solved = flat * pos
        solvedimg = img_blank.copy()
        sovedimg = displayNumber(solvedimg, solved)
        # sovedimg = displayNumber(solvedimgfinal, solved,(0,255,0))
        #
        #         ##overlay
        pts2 = np.float32(biggest)  # points for warp
        pts1 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgINwarmped = img.copy()
        imgINwarmped = cv2.warpPerspective(sovedimg, matrix, (width_img, height_img))
        # imgwarpcolored = cv2.cvtColor(imgwarpcolored, cv2.COLOR_BGR2GRAY)
        inv_perspective = cv2.addWeighted(imgINwarmped, 1, img, 0.5, 1)
        #img.save('/depoyment/static/image.jpg')
        # imgdetecteddigits = drawGrid(imgdetecteddigits)
        # solvedimg = drawGrid(solvedimg)
    # return imgINwarmped

    else:
        print('no sudoku found')
        #
        #
        # def display_image():
        #     cv2.imshow('original', img)
        #     cv2.imshow('threshold', img_threshold)
        #     cv2.imshow('contours', imgContours)
        #     cv2.imshow('bigcontour', imgBigcontour)
        #     cv2.imshow('warp', imgwarpcolored)
        #     #cv2.imshow('1', boxes[4])
        #     cv2.imshow('2',imgdetecteddigits)
    # cv2.imshow('3', sovedimg)
    # cv2.imshow('4', imgINwarmped)
    # cv2.imshow('5', inv_perspective)

    #
    #     #frame = overlay(frame, output)
    # cv2.imshow('pred', inv_perspective)
    #
   # cv2.waitKey(0)

    return inv_perspective

def delete():
    dir = "/depoyment/static"
    lists = os.listdir(dir)
    for item in lists:
        if item.endswith(".jpg"):
            os.remove(os.path.join(dir,item))

@app.route('/')
#@app.route('/home')
def home():
    delete()
    return render_template('main.html')
    #return('hi')



@app.route('/digit_recognition',methods =['POST','GET'])
def upload_image():
    if request.method=='POST':
        if 'file1' not in request.files:
            print( 'no file part')
        if request.files== " ":
            print('no file found',"danger")
        img = request.files['file1']
        img_name = secure_filename(img.filename)
        img.save('./static/image.jpg')
        #imgage = img.filename
        prediction = predict()
        #return redirect(url_for('predict'))
        return make_response(render_template('predict.html', predict=prediction, filename='image.jpg'))
    return render_template('upload.html')

# #@app.route('/predict',methods=['POST','GET'])
# @app.route('/capture', methods=['POST','GET'])
# def capture():
#     #pass a response cptured by camera to the webpage

    # return render_template('capture.html')
@app.route('/sudoku',methods =['POST','GET'])
def upload_images():
    if request.method=='POST':
        if 'file2' not in request.files:
            print( 'no file part')
        if request.files== " ":
            print('no file found',"danger")
        img = request.files['file2']
        img_name = secure_filename(img.filename)
        img.save('./static/sudoku.jpg')
        #imgage = img.filename
        prediction = predict_sudoku()
        #prediction.save('/depoyment/static/image1.jpg')
        cv2.imwrite('./static/solved.jpg',prediction)
        #return redirect(url_for('predict'))
        return render_template('predict1.html')
    return render_template('upload1.html')






@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('depoyment/static', filename='depoyment/static/image.jpg' , code=301))
if __name__ == '__main__':
    app.run(debug=True)

