
from detect.Detection import ObjectDetection
import os
from PIL import Image
import numpy as np

execution_path = os.getcwd()

i=1          ## if the file names are numeric and in sequence set i value to the 1st file name.  Eg: 1.jpg --> i = 1

num = 25     ##defune the threshold for the coordinates to expand on each side(left, top,  )

path = "pictures/"   ##path to the input images

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()         ##select yolo as a pre-trained model
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))   ##path to the yolo weight file
detector.loadModel()

while (i<=1):   ## range of the toltal files/images in the directory
    filename = path + "frame%d.jpg"%i
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , filename), output_image_path=os.path.join(execution_path , "output/frame%d.jpg"%i), display_object_name = False, display_percentage_probability = False, minimum_percentage_probability=90)
    for eachObject in detections:

        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        if eachObject["name"] == "person":

            xy = []
            xy.append(eachObject["box_points"])
            x1 = np.array(xy[0][0]) - num
            y1 = np.array(xy[0][1]) - num
            x2 = np.array(xy[0][2]) + num
            y2 = np.array(xy[0][3]) + num

            cnew=(x1,y1,x2,y2)      ##expanded coordinates
            #print("------->"+str(cnew))
            #print (eachObject["name"])

            img = Image.open("output/frame%d.jpg"%i)
            image_new = img.crop(cnew)    ##crop the object
            image_new.save("output/frame%d.jpg"%i)   ##output path
        print("--------------------------------")
    i += 1


