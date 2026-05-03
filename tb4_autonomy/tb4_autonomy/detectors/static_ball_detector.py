from __future__ import annotations
import cv2 as cv
import numpy as np
from tb4_autonomy.data_types import BallDetection, Box2D

class StaticObstacleDetector:
    name = "static_obstacle"

    def __init__(self): #(self,robot_speed_mps=0.15):
        # stores previous frame + turtlebot speed
        self.prev_gray = None
        #self.robot_speed_mps = robot_speed_mps

        # fixed distance reference for TTC (meters)
        #self.obstacle_ref_dist_m = 

    # detects round object ( ball-like)
    def detect(self, frame, context=None):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # image change to gray scale
         
        # frame 1 has no previous reference 
        if self.prev_gray is None:
            self.prev_gray = gray
            return None # no detection return none 
        
        # Computes optical flow using Farneback
        flow = cv.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
         
        # updates previous frame
        self.prev_gray = gray
        
        # computes by how much each pixels moves
        fx = flow[:, :, 0]
        fy = flow[:, :, 1]
        mag = np.sqrt(fx**2 + fy**2) # computes motion magnitude per pixel
        
        # smooth segements for stable optical flow
        mag_blur = cv.GaussianBlur(mag, (15, 15), 0)

        mean_mag = np.mean(mag_blur)
        std_mag = np.std(mag_blur)

        # highlights strong motion only
        thresh = mean_mag + 1.5 * std_mag
        
        # binary mask for moving regions
        mask = (mag_blur > thresh).astype(np.uint8) * 255
         
        # filters noise using OPEN + CLOSE morphological operation
        kernel = np.ones((7, 7), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=3)

        # normalizes optical flow magnitude
        mag_norm = cv.normalize(mag_blur, None, 0, 255, cv.NORM_MINMAX)
        mag_uint8 = mag_norm.astype(np.uint8)
        flow_color = cv.applyColorMap(mag_uint8, cv.COLORMAP_JET) # converts flow magnitude to colored map

        # shows colors only when high optical flow exists
        flow_view = np.zeros_like(frame)
        flow_view[mask > 0] = flow_color[mask > 0]

        # finds object contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        
        # loop through detected regions
        for cnt in contours:
            area = cv.contourArea(cnt) # computes size of the shape
            if area < 1500:
                continue    # skips small noise blobs

            perimeter = cv.arcLength(cnt,True) # computes the permiter of the shape
            if perimeter == 0: # checks and skips  invalid shape
                continue
            
            # isolates the ball from other shapes
            # checks for circuilarity, perfect circle has 1 circularity ( C^2 = 4*pi*Area)
            circularity = 4 * np.pi * area / (perimeter**2) 
            if circularity < .6: # filters shapes based on roundness
                continue
            x, y, w, h = cv.boundingRect(cnt) # bounding box

            # checks for bounding box (square-like boxes)
            aspect = w / float(h) 
            if aspect < 0.7 or aspect > 1.2: # filters based on shape proportions
                continue

            return BallDetection(
                box=Box2D(x, y, w, h), 
                moving=False, # marks as static object
                confidence=1.0
)
        return None