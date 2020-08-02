import cv2

#import  our car images / import vidéo
img_file = 'car.jpg'
#video = cv2.VideoCapture("tesla.mp4")
#video = cv2.VideoCapture("tesla.mp4")
video = cv2.VideoCapture("pedestrians.mp4")

#import our pre-trained car and pedestrian classifier
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'pedestrians_detector.xml'

#create car and pedestrian classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


#Run until car crashes
while True:
    #Read the frame from the video
    #read_successful = boolean
    #frame = array
    (read_successful, frame) = video.read()
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars and pedestrian
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #Draw a red rectangle around the car
    for (x, y, w, h) in cars : 
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) #blue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) #red

    #Draw a yellow rectangle around the pedestrian
    for (x, y, w, h) in pedestrians : 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    #Display the image with the face spotted
    cv2.imshow("Car detector", frame)

    key = cv2.waitKey(1)

    #stop if Q key is pressed
    if key == 81 or key == 113:
        break
#create opencv image
#img = cv2.imread(img_file)

#convert to grayscale ( convert color to black and white)
#black_n_white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#create car classifier
#car_tracker = cv2.CascadeClassifier(classifier_file)

#detect cars in the image
#cars = car_tracker.detectMultiScale(black_n_white)

#draw rectangles around the cars
#for (x, y, w, h) in cars : 
#    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#car1 = cars[7]
#(x, y, w, h) = car1

#display the image with the car spotted
#cv2.imshow('Car Detector',img)

#Don't autoclose (wait here in the code and listen for a key press)
#cv2.waitKey()

########## Test ##########
print("Code Completed")