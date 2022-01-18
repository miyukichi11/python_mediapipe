import os
import shutil
import cv2
import glob
import mediapipe as mp
import argparse
import numpy as np

mp_holistic = mp.solutions.holistic

# Initialize MediaPipe Holistic.
holistic = mp_holistic.Holistic(
    static_image_mode=True, min_detection_confidence=0.5)

# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 255, 0))

def make_images(video_file,   
                   image_dir, 
                   image_file="%s.png"):
    print("run make_images()")
 
    # Initial setting
    i = 0
    interval = 3
    length = 300
    
    cap = cv2.VideoCapture(video_file)
    while(cap.isOpened()):
        flag, frame = cap.read()  
        if flag == False:  
                break
        if i == length*interval:
                break
        if i % interval == 0:    
           cv2.imwrite(image_dir+image_file % str(int(i/interval)).zfill(6), frame)
        i += 1 
    cap.release()
    print("finish make_images()")
    
def image_write(image_dir):
    print("run image_write()")
    # image file names to files in list format
    files=[]
    for name in sorted(glob.glob(image_dir+"*.png")):
        files.append(name)
    #print(files)

    # Read images with OpenCV.
    images = {name: cv2.imread(name) for name in files}

    for name, image in images.items():
        #print(name, image)
        # Convert the BGR image to RGB and process it with MediaPipe Pose.
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw pose landmarks.
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image=annotated_image, 
            landmark_list=results.face_landmarks, 
            connections=mp_holistic.FACEMESH_TESSELATION,  ###
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
        mp_drawing.draw_landmarks(
            image=annotated_image, 
            landmark_list=results.pose_landmarks, 
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
        img_rotate_90_clockwise = cv2.rotate(annotated_image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(name, img_rotate_90_clockwise)
    print("finish image_write()")

def track_image_write(image_dir):
    print("run image_write()")
    # image file names to files in list format
    files=[]
    for name in sorted(glob.glob(image_dir+"*.png")):
        img = cv2.imread(name)
        height, width, layers = img.shape[:3]
        files.append(name)
    #print(files)
    
    blank = np.zeros((height, width, 3))
    blank += 255

    # Read images with OpenCV.
    images = {name: cv2.imread(name) for name in files}

    for name, image in images.items():
        #print(name, image)
        # Convert the BGR image to RGB and process it with MediaPipe Pose.
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw pose landmarks.
        #annotated_image = image.copy()
        mp_drawing.draw_landmarks(blank, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(blank, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image=blank, 
            landmark_list=results.face_landmarks, 
            connections=mp_holistic.FACEMESH_TESSELATION,  ###
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
        mp_drawing.draw_landmarks(
            image=blank, 
            landmark_list=results.pose_landmarks, 
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
        img_rotate_90_clockwise = cv2.rotate(blank, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(name, img_rotate_90_clockwise)

    print("finish image_write()")

def make_video(image_dir, output_filename):
    print("run make_video()")
    # image file names to img_files in list format
    img_files = []
    for name in sorted(glob.glob(image_dir+"*.png")):
        img = cv2.imread(name)
        if img is None: 
            print( "error! :no image") 
        else:
            height, width, layers = img.shape[:3]
            size = (width, height)
            #print(size)
            img_files.append(img)
    
    frame_rate = 10
    output_file = output_filename
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v') 
    output_video = cv2.VideoWriter( output_file, fourcc, frame_rate, size)

    for image in img_files:
        if image is None: 
            print( "error! :no image") 
        else: 
            output_video.write( image )

    output_video.release()
    
    print("finish make_video()")
    print("output file! : %s" % output_file)

def set_dir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="mediapipeのサンプルです。") 
    parser.add_argument("arg1", type=str, help="videoのPATHを指定してください。")
    parser.add_argument("--track", action="store_true", help="trackingを出力する場合は　--track をつけてください。")
    args = parser.parse_args() 
    print("inpit video : %s" % args.arg1)
    print("trackingの出力有無 : %d" % args.track)
    
    video_path = args.arg1
    image_path = "images/"
    simple_image_path = "tracking_images/"
    
    output_file = "output.mp4"
    
    set_dir(image_path)    
    make_images(video_file = video_path,image_dir = image_path)
    image_write(image_path)    
    make_video(image_path, output_file)
    
    if args.track==True:
        output_file = "track_output.mp4"
        set_dir(simple_image_path)    
        make_images(video_file = video_path,image_dir = simple_image_path)
        track_image_write(simple_image_path)    
        make_video(simple_image_path, output_file)


if __name__ == '__main__':
    main()