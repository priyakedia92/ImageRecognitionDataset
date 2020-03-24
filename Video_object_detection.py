import numpy as np
import cv2
import time

root_file_path="C:/Priya/Image Recognition Project/object-detection-opencv-master/"

IMAGE=root_file_path+"restaurant.jpg"

#Image=IMAGE

CLASSES=root_file_path+"yolov3.txt"

CONFIG=root_file_path+"yolov3.cfg"

WEIGHTS=root_file_path+"yolov3.weights"

scale=0.00392
conf_threshold=0.5
nms_threshold=0.4

classes=None

with open(CLASSES,"r") as f:
    classes=[line.strip() for line in f.readlines()]
    
COLORS=np.random.uniform(0,255,size=(len(classes),3))

def get_output_layers(net):
    layer_names=net.getLayerNames()
    output_layers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_bounding_boxes(img,class_id,x,y,x_plus_w,y_plus_h):
    label=str(classes[class_id])
    color=COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
#    cv2.rectangle(img,(x,y),(x_plus_w,y_plus_h),color,2)
    cv2.putText(img,label,(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,color,2)
    
def preprocess_image(Image,index,Height,Width):
#    image=cv2.imread(Image)
    image=Image
    width=Width
    height=Height
    
    net=cv2.dnn.readNet(WEIGHTS,CONFIG)
    
    blob=cv2.dnn.blobFromImage(image,scale,(416,416),(0,0,0),swapRB=True,crop=False)
    net.setInput(blob)
    
    outs=net.forward(get_output_layers(net))
    
    class_ids=[]
    confidences=[]
    boxes=[]
    count=1
#
#len(outs)
#len(outs[0])
#len(outs[1])
#len(outs[2])
   
    for out in outs:
        for detection in out:
            count+=1
            scores=detection[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:
                print(confidence)
                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)
                x=center_x-w/2
                y=center_y-h/2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x,y,w,h])
                
    print(count)
    
#    boxes=[[462.5, 88.0, 229, 72], [155.0, 122.0, 414, 306], [122.0, 223.0, 190, 322]]
#    confidences=[0.80841064453125, 0.9928321838378906, 0.9962299466133118]
#    
#    boxes=[[472.0, 86.5, 220, 79], [117.0, 124.5, 452, 307], [122.5, 223.0, 197, 320]]
#    confidences=[0.9397407174110413, 0.9899010062217712, 0.9979029893875122]
#                
    indices=cv2.dnn.NMSBoxes(boxes,confidences,conf_threshold,nms_threshold)
    
    for i in indices:
        i=i[0]
        box=boxes[i]
        x=box[0]
        y=box[1]
        w=box[2]
        h=box[3]
        print(x,y,w,h)
        
        draw_bounding_boxes(image,class_ids[i],round(x),round(y),round(x+w),round(y+h))
        
    return image
        
#    out_image_name="object detection final"+str(index)
#    cv2.imwrite(root_file_path+out_image_name+".jpg",image)
    
vs=cv2.VideoCapture(root_file_path+"Rice Shopping at the Supermarket and Play Area Indoor Playground for children.mp4")

writer=None
(W,H)=(None,None)

try:
    prop=cv2.CAP_PROP_FRAME_COUNT
    total=int(vs.get(prop))
    print("[INFO] {} Total Frames in Video".format(total))
except:
    print("Could not determine # of Frames in Video")

i=1          
while True:
#    print(i)
    (grabbed,frame)=vs.read()
#    i+=1
    
    if not grabbed:
        break
    
    if W is None or H is None:
        (H,W)=frame.shape[:2]
    start=time.time()
    output_frame=preprocess_image(frame,0,H,W)
    end=time.time()
        
    if writer is None:
        fourcc=cv2.VideoWriter_fourcc(*"MJPG")
        writer=cv2.VideoWriter(root_file_path+"VideoFileOutput.avi",fourcc,30,(frame.shape[1],frame.shape[0]),True)
        
        
    writer.write(output_frame)
elap=end-start 
       
print("[INFO] Single Frame took {:.4f} seconds".format(elap))
print("[INFO] Estimated total time to finish {:.4f} seconds".format(elap*total))
        
writer.release()
vs.release()
    