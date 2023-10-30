# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:48:53 2023

@author: Administrator
"""
import cv2
from ultralytics import YOLO
import AuxiliaryFunctions as af

def main():
#Settings
# Load the YOLOv8 model
    #model = YOLO('C:/Users/Administrator/runs/detect/yoloG/weights/best.pt')
    #model = YOLO('C:/Users/Administrator/runs/pose/yoloPoseA/weights/best.pt')
    model = YOLO('C:/Users/Administrator/runs/pose/yoloPoseWhite/weights/best.pt')
# Open the video file
    #video_path = "F:/PoseYolo/train/data/movies/M888.avi"
    #video_output = "F:/PoseYolo/train/data/movies/M888Yolo3.avi"
    #video_path = "F:/YaelShammaiData/BTBR 103-1m_cut.avi"
    video_path = "F:/YaelShammaiData/dataPoseWhite/BTBR 103-1m_cut.avi"
    video_output = "F:/YaelShammaiData/dataPoseWhite/BTBR 103-1mYoloposeW.avi"
    
########
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(video_output,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(frame,conf = 0.01,workers = 0, device = 0)
           # results = model.predict(frame,workers = 0, device = 0)
            try:
             object_results = af.AuxiliaryFunctions(frame, results, model)
             object_results.GetResults()
            # results = model.predict(frame,conf = 0.3)
            # Visualize the results on the frame
             for r in results:
                print(r.probs)
                print(r.boxes)
                print(r.keypoints)
                
            #annotated_frame = results[0].plot()
             annotated_frame= object_results.GetImage()
            except:
             annotated_frame = frame
            out.write(annotated_frame)

            # Break the loop if 'q' is pressed
           
            # Display the annotated frame
            #cv2.imshow("YOLOv8 Inference", annotated_frame)
            
            #cv2.waitKey(1) & 0xFF == ord("q"):
            #break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 