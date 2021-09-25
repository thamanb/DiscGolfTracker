import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

coords = []

# video to be analyzed
cap = cv2.VideoCapture('filenamehere')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))



first = True

# file to save new video to
out = cv2.VideoWriter('filenameHere.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

# function that will track a point on the body and draw a continuouse line
def draw(frame, results):
    global coords
    try:
            landmarks = results.pose_landmarks.landmark
    except:
        pass
    # type the body part you want to follow in all caps after
    # the .PoseLandmark. following the format on landmarks.png
    pointx = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x
    pointy = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y

    if len(coords) == 0:
        x1 = int(pointx * frame_width)
        y1 = int(pointy * frame_height)
        x2 = x1
        y2 = y1
        coords.append(((x1,y1),(x2,y2)))
    else:
        x1 = coords[-1][1][0]
        y1 = coords[-1][1][1]
        x2 = int(pointx * frame_width)
        y2 = int(pointy * frame_height)
        
        coords.append(((x1,y1),(x2,y2)))
        for i in coords:
            frame = cv2.line(img=frame, pt1=(i[0]), pt2=(i[1]), color=(255,0,0), thickness=1, lineType=cv2.LINE_8, shift=0)
        x1 = x2
        y1 = y2
    
    return frame

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
        

        
        
        # funtion to draw the landmarks on the picture
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # comment out above function and uncomment draw to use draw function
        #draw(frame, results)
       
        cv2.imshow('video', frame)
        out.write(frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    out.release()
 



    
