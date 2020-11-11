import cv2
# import imutils
import glob, os
import numpy as np
import time

image = cv2.imread("/Users/sunnysahu/Desktop/composite.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = gray.shape[:2]

MIN_MATCH_COUNT = 3

start_time = time.time()

for file in glob.glob("/Users/sunnysahu/template_small.jpg"):
    template = cv2.imread(file, 0)
    
    patchSize = 20
    
    orb = cv2.ORB_create(edgeThreshold = patchSize, patchSize = patchSize)
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(gray, None)

    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
               table_number = 6,
               key_size = 12,    
               multi_probe_level = 1)
    search_params = dict(checks = 100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []

    for pair in matches:
        if len(pair) == 2:
            if pair[0].distance < 0.7*pair[1].distance:
                good.append(pair[0])

    print('len(good) ', len(good))
    print('match %03d, min_match %03d, kp %03d' % (len(good), MIN_MATCH_COUNT, len(kp1)))
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = template.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        # dst contains points of bounding box of template in image. 
        # draw a close polyline around the found template:
        image = cv2.polylines(image,[np.int32(dst)], 
                              isClosed = True,
                              color = (0,255,0),
                              thickness = 5)
        
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               matchesMask = matchesMask, # draw only inliers
               flags = 2)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    if len(good)>MIN_MATCH_COUNT:
        output2 = cv2.drawMatches(template,kp1,gray,kp2,good,None,**draw_params)

    print('elapsed time ', time.time()-start_time)
    # cv2.imshow("Image", image)
    cv2.imshow('output_homography.jpg', image)
    cv2.imshow('output2.jpg', output2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()