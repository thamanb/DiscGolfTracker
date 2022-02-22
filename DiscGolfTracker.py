import cv2
import numpy as np
import mediapipe as mp




mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

drawCoords = []

wristCoords = []
hipCoords = []
shoulderCoords = []

# video to be analyzed
cap = cv2.VideoCapture('yourfilehere.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# file to save new video to
out = cv2.VideoWriter('filename.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

# calculates angle between 3 landmarks
def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) -np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360-angle
    
    return angle

# function that will track a point on the body and draw a continuouse line
def draw(frame, results):
    global drawCoords
    try:
            landmarks = results.pose_landmarks.landmark
    except:
        pass
    # type the body part you want to follow in all caps after
    # the .PoseLandmark. following the format on landmarks.png
    pointx = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
    pointy = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y

    if len(drawCoords) == 0:
        x1 = int(pointx * frame_width)
        y1 = int(pointy * frame_height)
        x2 = x1
        y2 = y1
        drawCoords.append(((x1,y1),(x2,y2)))
    else:
        x1 = drawCoords[-1][1][0]
        y1 = drawCoords[-1][1][1]
        x2 = int(pointx * frame_width)
        y2 = int(pointy * frame_height)
        
        drawCoords.append(((x1,y1),(x2,y2)))
        for i in drawCoords:
            frame = cv2.line(img=frame, pt1=(i[0]), pt2=(i[1]), color=(255,0,0), thickness=1, lineType=cv2.LINE_8, shift=0)
        x1 = x2
        y1 = y2
    
    return frame

#function takes the x/y coords from the shoulders and right elbow
#and outputs either rounding or not rounding based on their angle
#switch rightelbow to left elbow for LHBH
def rounding(frame, results):
    try:
        landmarks = results.pose_landmarks.landmark
    except:
        pass
    leftShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    rightShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    rightelbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    angle = calculateAngle(leftShoulder, rightShoulder,  rightelbow)
    if angle >= 85:
        round = "Not Rounding"
    else:
        round = "Rounding"
    cv2.putText(frame,
        round,
        (100,100),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0,0,0),
        3)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:    
    while cap.isOpened():
        ret, frame = cap.read()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame.flags.writeable = False
        results = pose.process(frame)
        
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        #function to draw the landmarks on the picture
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        rounding(frame, results)

        cv2.imshow('video', frame)
        out.write(frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    out.release()







    
