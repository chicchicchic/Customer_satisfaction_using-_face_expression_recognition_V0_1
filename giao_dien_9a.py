from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from tkinter import Tk, Text, BOTH, W, N, E, S
from tkinter.ttk import Frame, Button, Label, Style
from tkinter import *
import os
import random
from subprocess import Popen
from PIL import ImageTk, Image
import tkinter.font as font
import cv2
import numpy as np
import time
import threading
import array as arr
import collections
from tkinter import *
import tkinter.font as font
from tkinter import *
from PIL import ImageTk, Image
import cv2


root = Tk()
# Create a frame

root.title('MapAwareness')
root.geometry('1160x510') # Size 200, 200
root['bg']='#002200'


root.grid()
lmain = Label(root)
lmain.grid()
lmain.place(x=10,y=10)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# classifier = load_model('TH4a.h5') #Alexnet
classifier = load_model('TH5.h5')   #Alexnet
class_labels = ['dislike', 'like', 'neutral']

li = []
cap = cv2.VideoCapture(0)

# fps = cap.get(cv2.CAP_PROP_POS_FRAMES)



def video_stream():
    ret, frame = cap.read()

    img_counter = 0
    gray = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    faces = face_classifier.detectMultiScale(gray, 1.3, 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (227, 227), interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            addLiArray = class_labels[preds.argmax()]

            # APPEND ELEMENTS INTO LIST
            li.append(addLiArray)

            print(li)
            if addLiArray == 'dislike':
                print("Image saved")
                file = 'I:/DO AN TONG HOP/DATH5/Pictures/Dislike/' + str(img_counter) + '.jpg'

                cv2.imwrite(file, frame)
                print("screenshot taken ")

            elif addLiArray == 'like':
                print("Image saved")
                file = 'I:/DO AN TONG HOP/DATH5/Pictures/Like/' + str(img_counter) + '.jpg'

                cv2.imwrite(file, frame)
                print("screenshot taken ")

            elif addLiArray == 'neutral':
                print("Image saved")
                file = 'I:/DO AN TONG HOP/DATH5/Pictures/Neutral/' + str(img_counter) + '.jpg'

                cv2.imwrite(file, frame)
                print("screenshot taken ")


            print("Độ dài của mãng là: ", len(li))

            # OUTPUT RESULT AFTER N SECONDS
            if len(li) % 12 == 0:

                # ELEMENT = MAX IN LIST
                collections.Counter(li).most_common(1)[0]
                o = max(set(li), key=li.count)
                print("Phản ứng nhiều nhất là", o)

                # Lưu hình vào 1 Folder

                top = Toplevel()
                top.title('Result of the DEMO "Đồ án tổng hợp"')
                top.geometry('1200x765')  # Size 200, 200
                top['bg'] = '#2B1B17'
                canvas = Canvas(top, width=630, height=475)
                canvas.pack()

                myFont = font.Font(family='Helvetica', size=45, weight='bold')  # FONT of label "result"
                myFont1 = font.Font(family='Helvetica', size=20, weight='bold')  # FONT of result button
                myFont2 = font.Font(family='Helvetica', size=40, slant='italic', weight='bold')  # FONT of result button

                # OUTPUT RESULT IN GUI
                if o == 'dislike':
                    label0 = Label(top, fg="#0B615E", pady=5, font=myFont2, height=0, width=60,
                                   text="Result")
                    label0.pack()
                    img = ImageTk.PhotoImage(Image.open("Pictures/Dislike/0.jpg"))
                    canvas.create_image(0, 0, anchor=NW, image=img)
                    label = Label(top, fg="#0A2A12", pady=5, font=myFont, height=0, width=60,
                                  text="Your customers don't like your product...\n༼ಢ_ಢ༽")
                    label.pack()
                    f = open('text1.txt', mode='a+')
                    f.write('\nYour customers do not like your product.')
                    f.close()

                    li.clear()



                elif o == 'like':
                    label0 = Label(top, fg="#0B615E", pady=5, font=myFont2, height=0, width=60,
                                   text="Result")
                    label0.pack()
                    img = ImageTk.PhotoImage(Image.open("Pictures/Like/0.jpg"))
                    canvas.create_image(0, 0, anchor=NW, image=img)
                    label = Label(top, fg="#0A2A12", pady=5, font=myFont, height=0, width=60,
                                  text="Your customers like your product...\n(/◕ヮ◕)/")
                    label.pack()
                    f = open('text1.txt', mode='a+')
                    f.write('\nYour customers like your product.')
                    f.close()

                    li.clear()

                elif o == 'neutral':

                    label0 = Label(top, fg="#0B615E", pady=5, font=myFont2, height=0, width=60,
                                   text="Result")
                    label0.pack()
                    img = ImageTk.PhotoImage(Image.open("Pictures/Neutral/0.jpg"))
                    canvas.create_image(0, 0, anchor=NW, image=img)
                    label = Label(top, fg="#0A2A12", pady=5, font=myFont, height=0, width=60,
                                  text="Your customers doesn't respond...\n(¯―¯٥)")
                    label.pack()

                    # SAVE TO FILE.TXT
                    f = open('text1.txt', mode='a+')
                    f.write('\nYour customers does not respond.')
                    f.close()

                    # REFRESH LIST
                    li.clear()




                top.mainloop()
                break



            # TAKE A PICTURE AFTER EVERY N SECONDS
            time.sleep(10)


        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # resized_img = cv2.resize(frame, (400, 400))
    # cv2.imshow('Emotion Detector', frame)

    # ============================================

    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, video_stream)


        # ============================================



        # k = cv2.waitKey(20)
        #
        # if k == ord('q'):
        #     break

    # cap.release()
    # cv2.destroyAllWindows()



def stop():
    print ("Stop")
    root.destroy()


# FONT
myFont = font.Font(family='Helvetica', size=40, weight='bold')
# Detect Button (Webcam)
videoButton = Button(root, font = myFont,  text ="📷  Recognition",bg='#FFFF00', fg="#222222",height=2, width=14,command = video_stream)
# Stop Button
stopButton = Button(root, font = myFont,  text ="❎        Exit      ",bg='#FF0000',fg="#FFFFCC",height=2, width=14,command = stop)



videoButton.place(x=680,y=40)
stopButton.place(x=680,y=260)


root.mainloop()



