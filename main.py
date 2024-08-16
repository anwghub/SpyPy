import cv2
import time
import datetime

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORDS_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))  #3:4 frame
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


while True:
    _,frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  #(it can give you multiple faces, scale-->(1.1 to 1.5, min number of neighbours -->(3-6))
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5) 

    if len(faces) + len(bodies) >0:
        if detection:
            timer_started= False
        else:
            detection = True
            currect_time = datetime.datetime.now(),time.strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{currect_time}.mp4", fourcc, 20, frame_size) #frame_rate = 20
            print("Started recording!")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORDS_AFTER_DETECTION:
                detection =False
                timer_started = False
                out.release()
                print("Stopped Recording")
        else:
            timer_started = True
            detection_stopped_time = time.time()

    out.write(frame)

    # for(x, y, width,height) in faces:
    #     cv2.rectangle(frame, (x,y), (x + width, y + height), (255,0,0), 3)


    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()