#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


img_array = cv.imread(r"C:\Users\Acer\Desktop\FaceMask\DataSet\face_Mask\00000_mask.jpg")


# In[3]:


#plt.imshow(img_array)


# In[4]:


#plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))


# In[5]:


img_array.shape


# In[6]:


Datadirectory = r"C:\Users\ASUS\Documents\assignments yr 3\New folder1\cw2\python\dataset"
Classes = ["with_mask", "without_mask"]
for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv.imread(os.path.join(path,img))
        plt.imshow(cv.cvtColor(img_array, cv.COLOR_BGR2RGB))
        plt.show()
        break
    break


# In[7]:


img_size =224

new_array= cv.resize(img_array, (img_size,img_size))
plt.imshow(cv.cvtColor(new_array, cv.COLOR_BGR2RGB))
plt.show()


# In[8]:


training_Data = []
def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path,img))
                new_array = cv.resize(img_array,(img_size, img_size))
                training_Data.append([new_array, class_num])
            except Exception as e:
                pass 
            


# In[9]:


#create_training_Data()


# In[12]:


#print(len(training_Data))


# In[13]:


import random
random.shuffle(training_Data)


# In[14]:


X= []
y= []
for features,label in training_Data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)


# In[15]:


X.shape


# In[16]:


X= X/255.0;


# In[17]:


Y= np.array(y)


# In[18]:


import pickle

#pickle_out = open("X.pickle","wb")
#pickle.dump(X, pickle_out)
#pickle_out.close()

#pickle_out = open("y.pickle","wb")
#pickle.dump(y, pickle_out)
#pickle_out.close()


# In[19]:


#pickle_in = open("X.pickle","rb")
#X = pickle.load(pickle_in)

#pickle_in = open("y.pickle","rb")
#y = pickle.load(pickle_in)


# In[20]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[21]:


model = tf.keras.applications.mobilenet.MobileNet()


# In[22]:


base_input = model.layers[0].input


# In[23]:


base_output= model.layers[-4].output


# In[24]:


Flat_layer= layers.Flatten()(base_output)
final_output = layers.Dense(1)(Flat_layer)
final_output = layers.Activation('sigmoid')(final_output)


# In[25]:


#new_model = keras.Model(inputs = base_input, outputs= final_output)


# In[26]:



#new_model.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])



# In[27]:


#new_model.fit(X,Y, epochs = 1,validation_split = 0.1)


# In[28]:


#new_model.save("my_model_cw.h5")


# In[23]:


new_model = tf.keras.models.load_model('my_model_cw.h5')


# In[30]:


frame = cv.imread('00013_Mask.jpg')


# In[ ]:





# In[31]:


plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))


# In[32]:


final_image = cv.resize(frame, (224,224))
final_image = np.expand_dims(final_image,axis =0)
final_image=final_image/255.0


# In[33]:


predections =new_model.predict(final_image)


# In[34]:


predections


# In[12]:


frame = cv.imread(r'C:\Users\Acer\Desktop\FaceMask\DataSet\No_Mask\00005.png')


# In[13]:


plt.imshow(frame)


# In[14]:


faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default')


# In[ ]:





# In[15]:


gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


# In[17]:


gray.shape


# In[18]:


#faces = faceCascade.detectMultiScale(gray,1.1,4)
#for x,y,w,h in faces:
   # roi_gray = gray[y:y+h, x:x+w]
   # roi_color = frame[y:y+h, x:x+w]
   # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0),2)
   # facess = facecascade.detectMultiScale(roi_gray)
   # if len(facess) == 0:
   #     print("face not detected")
   # else:
   #     for (ex,ey,ew,eh) in faces:
   #         face_roi = roi_color[ey: ey+eh, ex:ex + ew]


# In[26]:


    import cv2 as cv
    path = "haarcascade_frontalface_default.xml"
    font_scale = 1.5
    font = cv.FONT_HERSHEY_PLAIN



    rectangle_bgr = (255, 255, 255)

    img = np.zeros((500,500))

    text = "Detecting"

    (text_width, text_height) = cv.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    test_offset_x = 10
    test_offset_y = img.shape[0]-25
    box_coords =((test_offset_x, test_offset_y), (test_offset_x + text_width + 2, test_offset_y - text_height -2))
    cv.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv.FILLED)
    cv.putText(img, text, (test_offset_x, test_offset_y), font, fontScale=font_scale, color=(0,0,0), thickness=1)

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("cannot open webcam")

    while True:
        ret,frame = cap.read()

        faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(gray,1.1,4)
        
        for x,y,w,h in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            facess = faceCascade.detectMultiScale(roi_gray)
            if len(facess) == 0:
                print("face not detected")
            else:
                for (ex,ey,ew,eh) in facess:
                    
                 face_roi = roi_color[ey: ey+eh, ex:ex + ew]
        
       
        final_image = cv.resize(face_roi, (244,244))
        final_image = np.expand_dims(final_image, axis = 0)
        final_image = final_image/255.0
        font = cv.FONT_HERSHEY_SIMPLEX
        predections = new_model.predict(final_image)
        
        font_scale = 1.5
        font = cv.FONT_HERSHEY_PLAIN
        
        if (predections>0):
            status = "No Mask"
            
            x1,y1,w1,h1 = 0,0,175,75
            
            cv.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
            
            cv.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
            cv.putText(frame,status,(100, 150),font, 3,(0, 0, 255),2,cv.LINE_4)
            
            cv.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))
            print(" Face Mask = False")
            
        else:
            status = "Face Mask"
            
            x1,y1,w1,h1 = 0,0,175,75
            
            cv.rectangle(frame, (x1,x1),(x1 + w1, y1 + h1), (0,0,0), -1)
            
            cv.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            cv.putText(frame,status,(100, 150),font, 3,(0, 255, 0),2,cv.LINE_4)
            
            cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0))
            
            cv.imshow('Shubhangi Piya', frame)
            print("Face Maks= True")
            
            
            if cv.waitKey(2) & 0xFF == ord('q'):
               break
    cap.release()
    cv.destroyAllWindows()


    # In[60]:


    #plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))


    # In[25]:





    # In[ ]:





