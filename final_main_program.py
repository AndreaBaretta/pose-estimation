#import appropriate python modules to the program
import sys
import numpy as np
import cv2
import pickle
from matplotlib import pyplot as plt

print(cv2.__version__)

maxIterations = 20

solve_pnp_flag=cv2.SOLVEPNP_ITERATIVE
def imshow(w, f):
    cv2.imshow(w, f)

# capturing video from Kinect Xbox 360
def get_video(vc):
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        imshow("raw_frame", frame)
        return frame
        #array = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #return array
    else:
        return None

# callback function for selecting object by clicking 4-corner-points of the object
def select_object(event, x, y, flags, param):
    global box_pts, frame
    if input_mode and event == cv2.EVENT_LBUTTONDOWN and len(box_pts) < 4:
        box_pts.append([x, y])
        frame = cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)

# selecting object by clicking 4-corner-points
def select_object_mode():
    global input_mode, initialize_mode, frame_static
    input_mode = True

    while len(box_pts) < 4:
        imshow("frame", frame)
        cv2.waitKey(1)
    
    initialize_mode = True
    input_mode = False

# setting the boundary of reference object
def set_boundary_of_reference(box_pts):
    
    ### upper bound ###
    if box_pts[0][1] < box_pts[1][1]:
        upper_bound = box_pts[0][1]
    else:
        upper_bound = box_pts[1][1]
    
    ### lower bound ###
    if box_pts[2][1] > box_pts[3][1]:
        lower_bound = box_pts[2][1]
    else:
        lower_bound = box_pts[3][1]
    
    ### left bound ###
    if box_pts[0][0] < box_pts[2][0]:
        left_bound = box_pts[0][0]
    else:
        left_bound = box_pts[2][0]
    
    ### right bound ###
    if box_pts[1][0] > box_pts[3][0]:
        right_bound = box_pts[1][0]
    else:
        right_bound = box_pts[3][0]
        
    upper_left_point = [0,0]
    upper_right_point = [(right_bound-left_bound),0]
    lower_left_point = [0,(lower_bound-upper_bound)]
    lower_right_point = [(right_bound-left_bound),(lower_bound-upper_bound)]
    
    pts2 = np.float32([upper_left_point, upper_right_point, lower_left_point, lower_right_point])
    
    # display dimension of reference object image to terminal
    print(pts2)
    
    return pts2, right_bound, left_bound, lower_bound, upper_bound

# doing perspective transform to reference object
def input_perspective_transform(box_pts, pts2, right_bound, left_bound, lower_bound, upper_bound):
    global object_orb
    pts1 = np.float32(box_pts)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_object = cv2.warpPerspective(frame,M,((right_bound-left_bound),(lower_bound-upper_bound)))
    cv2.imshow("img_object", img_object)
    return cv2.cvtColor(img_object, cv2.COLOR_BGR2GRAY)

# feature detection and description using ORB
def orb_feature_descriptor(img_object):
    kp1, des1 = orb.detectAndCompute(img_object,None)
    kp2, des2 = orb.detectAndCompute(frame,None)
    return kp1, des1, kp2, des2

# feature matching using Brute Force
def brute_force_feature_matcher(kp1, des1, kp2, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    #return sorted(matches, key = lambda x:x.distance)
    sorted_matches = sorted(matches, key = lambda x:x.distance)
    #print "# of matches =", len(sorted_matches)
    #returning the top 120 matches
    return sorted_matches[:80]

# finding homography matrix between reference and image frame
def find_homography_object(kp1, kp2, matches):
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    return M, mask

# applying homography matrix as inference of perpective transformation
def output_perspective_transform(img_object, M):
    h,w = img_object.shape
    corner_pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    center_pts = np.float32([ [w/2,h/2] ]).reshape(-1,1,2)
    corner_pts_3d = np.float32([ [-w/2,-h/2,0],[-w/2,(h-1)/2,0],[(w-1)/2,(h-1)/2,0],[(w-1)/2,-h/2,0] ])###
    corner_camera_coord = cv2.perspectiveTransform(corner_pts,M)###
    center_camera_coord = cv2.perspectiveTransform(center_pts,M)
    return corner_camera_coord, center_camera_coord, corner_pts_3d, center_pts

# solving pnp
def solve_pnp(object_points, image_points, prev_rotation, prev_translation):
    image_points = image_points.reshape(-1,2)
    #image_points = np.ascontiguousarray(image_points[:,:2]).reshape(image_points[0],1,2)

    use_prev_guess = prev_rotation is not None and prev_translation is not None
    

    retval, rotation, translation = cv2.solvePnP(object_points, image_points, intrinsic_param, distortion_param, flags=solve_pnp_flag)

    axis = np.float32([[20,0,0], [0,20,0], [0,0,-20]]).reshape(-1,3)
    
    img_pts, jac = cv2.projectPoints(axis, rotation, translation, intrinsic_param, distortion_param)

    
    def draw(img_, corners, imgpts):
        corner = tuple(corners[0].ravel())
        print("corner =", corner)
        print("imgpts =",imgpts)
        print("ravel =", imgpts[0].ravel())
        corner = corner[0:2]
        img_ = cv2.line(img_, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img_ = cv2.line(img_, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img_ = cv2.line(img_, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img_

    
    reprojected = draw(frame, object_points, img_pts)
    cv2.imshow('reprojected',reprojected)


    #debug by reprojecting points
    
    #if use_prev_guess:
    #    retval, rotation, translation = \
    #        cv2.solvePnP(object_points, image_points, intrinsic_param, distortion_param, flags=solve_pnp_flag)
    #else:
    #    retval, rotation, translation = cv2.solvePnP(object_points, image_points, intrinsic_param, distortion_param, flags=solve_pnp_flag)
        
        
        
    return rotation, translation

# drawing box around object
def draw_box_around_object(dst):
    return cv2.polylines(frame, [np.int32(dst)],True,255,3)
    
# recording sample data
def record_samples_data(translation, rotation):
    translation_list = translation.tolist()
    rotation_list = rotation.tolist()
    
    t1.append(translation_list[0])
    t2.append(translation_list[1])
    t3.append(translation_list[2])
    
    r1.append(rotation_list[0])
    r2.append(rotation_list[1])
    r3.append(rotation_list[2])
    
# computing and showing recorded data to terminal
def showing_recorded_data_to_terminal(t1, t2, t3, r1, r2, r3):
    
    # convert to numpy array
    t1 = np.array(t1)
    t2 = np.array(t2)
    t3 = np.array(t3)
    
    r1 = np.array(r1)
    r2 = np.array(r2)
    r3 = np.array(r3)
    
    # print mean and std of the data to terminal
    print("mean t1", np.mean(t1))
    print("std t1", np.std(t1))
    print("")
    print("mean t2", np.mean(t2))
    print("std t2", np.std(t2))
    print("")
    print("mean t3", np.mean(t3))
    print("std t3", np.std(t3))
    print("")
    print("")
    print("mean r1", np.mean(r1))
    print("std r1", np.std(r1))
    print("")
    print("mean r2", np.mean(r2))
    print("std r2", np.std(r2))
    print("")
    print("mean r3", np.mean(r3))
    print("std r3", np.std(r3))
    print("")
    print("#####################")
    print("")

# showing object position and orientation value to frame
def put_position_orientation_value_to_frame(_translation, _rotation):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    translation = np.transpose(_translation)[0]
    rotation = np.transpose(_rotation)[0]
    
    
    print("translation:", translation)
    print("rotation:", rotation)
    
    cv2.putText(frame,'position(cm)',(10,30), font, font_scale,(0,255,0),font_thickness,cv2.LINE_AA)
    cv2.putText(frame,'x:'+str(round(translation[0],2)),(250,30), font, font_scale,(0,0,255),font_thickness,cv2.LINE_AA)
    cv2.putText(frame,'y:'+str(round(translation[1],2)),(350,30), font, font_scale,(0,0,255),font_thickness,cv2.LINE_AA)
    cv2.putText(frame,'z:'+str(round(translation[2],2)),(450,30), font, font_scale,(0,0,255),font_thickness,cv2.LINE_AA)
    
    cv2.putText(frame,'orientation(degree)',(10,60), font, font_scale,(0,255,0),font_thickness,cv2.LINE_AA)
    cv2.putText(frame,'x:'+str(round(rotation[0],2)),(250,60), font, font_scale,(0,0,255),font_thickness,cv2.LINE_AA)
    cv2.putText(frame,'y:'+str(round(rotation[1],2)),(350,60), font, font_scale,(0,0,255),font_thickness,cv2.LINE_AA)
    cv2.putText(frame,'z:'+str(round(rotation[2],2)),(450,60), font, font_scale,(0,0,255),font_thickness,cv2.LINE_AA)
    
    return frame


############
### Main ###
############

# initialization
input_mode = False
initialize_mode = False
track_mode = False
box_pts = []
pickle_mode = False

record_num = 0
record_mode = False

t1, t2, t3, r1, r2, r3 = [], [], [], [], [], []

with open('calib.pickle','rb') as picklefile:
    ret, mtx, dist, rvecs, tvecs = pickle.load(picklefile)
    intrinsic_param = mtx
    distortion_param = dist
    print("mean t1", np.mean(t1))
    print("std t1", np.std(t1))
    print("")
    print("mean t2", np.mean(t2))
    print("std t2", np.std(t2))
    print("")
    print("mean t3", np.mean(t3))
    print("std t3", np.std(t3))
    print("")
    print("")
    print("mean r1", np.mean(r1))
    print("std r1", np.std(r1))
    print("")
    print("mean r2", np.mean(r2))
    print("std r2", np.std(r2))
    print("")
    print("mean r3", np.mean(r3))
    print("std r3", np.std(r3))
    print("")
    print("#####################")
    print("")

#kinect_intrinsic_param = np.array([[514.04093664, 0., 320], [0., 514.87476583, 240], [0., 0., 1.]])
#kinect_distortion_param = np.array([2.68661165e-01, -1.31720458e+00, -3.22098653e-03, -1.11578383e-03, 2.44470018e+00])

orb = cv2.ORB_create()

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", select_object)

vc  = cv2.VideoCapture(0)


rotation = None
translation = None

if '--use-pickle' in sys.argv:
    with open('reference_setup.pickle', 'rb') as pickle_file:
        (frame_static, pts2, right_bound, left_bound, lower_bound, upper_bound, img_object, kp1, des1) = pickle.load(pickle_file)
    pickle_mode = True
    output_frame = frame_static
    select_object_mode

while True:
    
    frame = get_video(vc)
    k = cv2.waitKey(1) & 0xFF

    #global frame_static #new line
    frame_static = frame.copy()

    if pickle_mode and k == ord('i'):
        track_mode = True
    
    if not track_mode:

        # press i to enter input mode
        if k == ord('i'):
            output_frame = frame.copy()
            # select object by clicking 4-corner-points
            select_object_mode()

            # set the boundary of reference object
            pts2, right_bound, left_bound, lower_bound, upper_bound = set_boundary_of_reference(box_pts)

            # do perspective transform to reference object
            img_object = input_perspective_transform(box_pts, pts2, right_bound, left_bound, lower_bound, upper_bound)
            kp1, des1 = orb.detectAndCompute(img_object,None)
            #print("# of ORB descriptors (static) =", len(des1))
            #print("# of ORB descriptors (static) =", len(kp1))

            #print("track_mode is being set to True")
            #with open('reference_setup.pickle', 'wb') as pickle_file:
            #    pickle.dump((frame_static, pts2, right_bound, left_bound, lower_bound, upper_bound, img_object, kp1, des1), pickle_file)
            track_mode = True
    
    # track mode is run immediately after user selects 4-corner-points of object
    else:
        # feature detection and description
        kp2, des2 = orb.detectAndCompute(frame,None)
        
        # feature matching
        matches = brute_force_feature_matcher(kp1, des1, kp2, des2)

        #draw match
        output_frame = cv2.drawMatches(frame_static, kp1, frame, kp2, matches, output_frame, flags=2)
        imshow("output_frame", output_frame)
        
        # find homography matrix
        M, mask = find_homography_object(kp1, kp2, matches)
        
        # apply homography matrix using perspective transformation
        corner_camera_coord, center_camera_coord, object_points_3d, center_pts = \
            output_perspective_transform(img_object, M)
        
        # solve pnp using iterative LMA algorithm
        rotation, translation = solve_pnp(object_points_3d, corner_camera_coord, rotation, translation)
        
        # convert to centimeters
        # translation = (40./53.) * translation *.1
        
        # convert to degree
        # rotation = rotation * 180./np.pi

        rotation_front, jacobian = cv2.Rodrigues(rotation)

        #print(type(rotation_front))
        #print(rotation_front)

        rotation_inverse = cv2.transpose(rotation_front)

        world_position_front_cam = -rotation_inverse*translation
        
        # press r to record 50 sample data and calculate its mean and std
        if k == ord("r"):
            record_mode = True
            
        if record_mode is True :
            record_num = record_num + 1
            
            # record 50 data
            record_samples_data(translation, rotation)
            
            if record_num == 50:
                record_mode = False
                record_num = 0
                
                # compute and show recorded data
                showing_recorded_data_to_terminal(t1, t2, t3, r1, r2, r3)
                
                # reset the data after 50 iterations
                t1, t2, t3, r1, r2, r3 = [], [], [], [], [], []
        
        # draw box around object
        frame = draw_box_around_object(corner_camera_coord)
        
        # show object position and orientation value to frame
        frame = put_position_orientation_value_to_frame(translation, rotation)

        print("world_position =", world_position_front_cam)

    imshow("frame", frame)
    #if track_mode != True and frame != None:
        #cv2.imshow("debug_frame", debug_frame)
    #else:
        #cv2.imshow("frame",frame)

    # break when user pressing ESC
    if k == 27:
        break

cv2.destroyAllWindows()
