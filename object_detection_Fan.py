# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2 # to draw the rectangles
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections
def detect(frame, net, transform):
    height, width = frame.shape[:2] # There are 3 parameters of frame.shape: height, width and color
    
    # Transform 1: to numpy array
    frame_t = transform(frame)[0] # transform return 2 elements
    
    # Transform 2: to torch tensor (more advanced matrix than numpy array)
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # to arrange the colors from RBG to GRB
    
    # Transform 3: to CNN acceptable batch format with the 4th dimension added
    x.unsqueeze(0) # dimension # stats from 0: we added the batch dimension to position 0
    
    # Transform4: to torch variable
    x = Variable(x.unsqueeze(0))
    
    # Feed our variable to the nueral network
    y = net(x) # the network will output
    
    # As the output contains both data and more, we need to pick the data part out
    detections = y.data
    
    # To normalize the detection data
    scale = torch.Tensor([width, height, width, height])
    ''' 
    the 1st width and height corresponds to the upper 
    left corner of the triangle
    the 2nd width and height corresponds to the lower 
    right conner of the triangle
    
    detections = [batch, number of classes, number of occurence, (score, x0, y0, x1, y1]
    class: for each class of objects, such as a bird
    number of occurence: how many times a class is detected
    score: thredhold for objection detection. If over a certain value, the detection will be valid.
    x0, y0: upper left corner of the detection triangle
    x1, y1: lower right corner of the dtection triagnel
    '''
    for i in range(detections.size(1)):
        j = 0 # number of occurence
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            
            # draw the rectangle
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) # We draw a rectangle around the detected object.
            '''
            cv2.rectangle(frame, upperleftpoint(x,y), lowerrightpoint(x,y), color, thickness)
            '''
            # display the labels
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) # We put the label of the class right above the rectangle.
            j += 1 # We increment j to get to the next occurrence.
    return frame # We return the original frame with the detector rectangle and the label around the detected object.
        
# Creating the SDD nuerual netowork
net = build_ssd('test')


# Load the previously trained weights
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).
# Creating the transforms
# 1st parameter: the figure size taken into the neural network
# 2nd parameter: the color range
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Doing some Object Detection on a video

#load the video
reader = imageio.get_reader('funny_dog.mp4')
#get the frame rate parameter
fps = reader.get_meta_data()['fps']

# output the video with detections
writer = imageio.get_writer('output.mp4', fps = fps)
for i, frame in enumerate(reader):
    # net.eval() is the function we want to pass into the detec function
    frame = detect(frame, net.eval(), transform)
    # now frame has the detection rectangles
    
    # To append a frame to the output video
    writer.append_data(frame)
    print(i)
# close this output video    
writer.close()
    