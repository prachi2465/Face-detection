import cv2
import time
import numpy as np
import csv
from imutils import paths
from datetime import datetime
from imutils.object_detection import non_max_suppression
import imutils
import time
import os
import math
img= 'C:\\Users\\prach\\Desktop\\python\\images'
lbp_face_cascade1=cv2.CascadeClassifier('C:\\Users\\prach\\Desktop\\python\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml')
lbp_face_cascade2=cv2.CascadeClassifier('C:\\Users\\prach\\Desktop\\python\\data\\lbpcascades\\lbpcascade_profileface.xml')
#haar_face_cascade1=cv2.CascadeClassifier('C:\\Users\\prach\\Desktop\\python\\data\\haarcascades\\haarcascade_frontalface_default.xml')
haar_face_cascade2=cv2.CascadeClassifier('C:\\Users\\prach\\Desktop\\python\\data\\haarcascades\\haarcascade_profileface.xml')
#scale_factor = 1.2
X= []
Y= []
count= 0
for imagePath in paths.list_images(img):
    test = cv2.imread(imagePath)
    #test= imutils.resize(test, width=min(900, test.shape[1]))
    orig= test.copy()
    faces1 = lbp_face_cascade1.detectMultiScale(test, 1.2, minNeighbors=5)
    faces2 = lbp_face_cascade2.detectMultiScale(test, 1.2, minNeighbors=5)
    if (len(faces2) and len(faces1))== 0:
        faces4 = haar_face_cascade2.detectMultiScale(test, 1.2, 3)
        for (x, y, w, h) in faces4:
            cv2.rectangle(orig, (x, y), (x+w, y+h), (255, 255, 255), 0)
        faces4= np.array([[x, y, x+w, y+h] for (x, y, w, h) in faces4])
        f4= non_max_suppression(faces4, probs= None, overlapThresh= 0.65)
        for (xa, ya, xb, yb) in f4:
            #cv2.rectangle(test, (xa, ya), (xb, yb), (255, 255, 255), 0)
            crp= test[ya:yb, xa:xb]
            cv2.imwrite(os.path.join('C:\\Users\\prach\\Desktop\\python\\humanDetection\\images', "facehaar_p%d.jpg" % count), crp)
            count+=1
            cv2.imshow(' ', crp)
        print("Number of profile faces{0}", format(len(f4)))
        #cv2.imshow('face',test)
        cv2.waitKey(300)
        b= len(f4)
    else:
        for (x, y, w, h) in faces1:
            #cv2.rectangle(test, (x, y), (x+w, y+h), (0, 255, 255), 0)
            crp= test[y:y+w, x:x+h]
            cv2.imwrite(os.path.join('C:\\Users\\prach\\Desktop\\python\\humanDetection\\images', "facelbp_f%d.jpg" % count), crp)
            count+=1
            cv2.imshow(' ', crp)
        print("Number of frontal faces{0}", format(len(faces1)))
        #cv2.imshow('face',test)
        cv2.waitKey(300)
        for (x, y, w, h) in faces2:
            #cv2.rectangle(test, (x, y), (x+w, y+h), (0, 255, 0), 0)
            crp= test[y-math.floor(h/4):y+h, x-math.floor(h/4):x+w]
            cv2.imwrite(os.path.join('C:\\Users\\prach\\Desktop\\python\\humanDetection\\images', "facelbp_p%d.jpg" % count), crp)
            count+=1
            cv2.imshow(' ', crp)
        print("Number of profile faces{0}", format(len(faces2)))
        #cv2.imshow('face',test)
        cv2.waitKey(300)
        b= len(faces2)
    X.insert(len(X), max(len(faces1), b))
    Y.insert(len(Y), str(datetime.now()))
    with open('data.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #spamwriter.writerow(['Number of people'] + ['Time'])
        spamwriter.writerow([max(len(faces1), b), str(datetime.now())+'!'])
print(X)
print(Y)
# importing the required module
import matplotlib.pyplot as plt
 
# plotting the points 
plt.plot(Y, X)
 
# naming the x axis
plt.xlabel('Time')
# naming the y axis
plt.ylabel('Number of people')
 
# giving a title to my graph
plt.title('')
 
# function to show the plot
plt.show()



