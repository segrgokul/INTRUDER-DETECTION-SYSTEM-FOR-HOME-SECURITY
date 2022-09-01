
import glob
from msilib.schema import Font
from platform import release
import cv2
import numpy as np
import time
import datetime
import face_recognition
from twilio.rest import Client
import smtplib
import imghdr
from email.message import EmailMessage
import glob

Sender_Email = "#your mail id 1"
Reciever_Email = "your mail id 2"
Password = "your game two verification password"

newMessage = EmailMessage()                         
newMessage['Subject'] = "Check out the inturder" 
newMessage['From'] = Sender_Email                   
newMessage['To'] = Reciever_Email                   
newMessage.set_content('someone is there at your gate. Image attached!') 


# Read text from the credentials file and store in data variable
with open('credentials.txt', 'r') as myfile:
  data = myfile.read()

# Convert data variable into dictionary
info_dict = eval(data)


def send_message(body, info_dict):
    
    # Your Account SID from twilio.com/console
    account_sid = info_dict['account_sid']

    # Your Auth Token from twilio.com/console
    auth_token  = info_dict['auth_token']


    client = Client(account_sid, auth_token)

    message = client.messages.create( to = info_dict['your_num'], from_ = info_dict['trial_num'], body= body)
    
start_time = time.time()
fps = 0
frame_counter = 0
status = False
patience =7
cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
# Get width and height of the frame
width = int(cap.get(3))
height = int(cap.get(4))

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = False
detection_stopped_time = None
timer_started = False
initial_time = None
frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

paths = glob.glob('C://Users//Admin//Desktop//IDS demo//data//*')
names = []
images = []
image_encodings = []
image_names = []
count_img = 0
print("Recognising faces...Encoding features....")
for i in paths:
    images.append(face_recognition.load_image_file(i))
    image_encodings.append(face_recognition.face_encodings(images[count_img])[0])
    image_names.append(i.split('\\')[-1].split('.')[0])
    count_img+=1
    print(image_names[-1], end=', ')
count = 0
while True:
    ret,frame = cap.read()
    frame_counter += 1
    fps = (frame_counter / (time.time() - start_time))
    
    current_time = datetime.datetime.now().strftime("%A, %I-%M-%S %p %d %B %Y")
    
      
    # Display the FPS
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (510, 450), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 40, 155),2)
    # Display the time
    cv2.putText(frame, current_time, (310, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),1)  
     # Display the Room Status
    cv2.putText(frame, 'Room Occupied: {}'.format(str(status)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (200, 10, 150),2)
      
    gray = frame[:, :, ::-1]

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = face_cascade.detectMultiScale(gray, 1.3, 5)
   

    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            status = True
            entry_time = datetime.datetime.now().strftime("%A, %I-%M-%S %p %d %B %Y")
            out = cv2.VideoWriter('C:/Users/Admin/Desktop/IDS demo/outputs/{}.mp4'.format(entry_time), cv2.VideoWriter_fourcc(*'XVID'), 3.5, (width, height))
            print("Started Recording!")
            initial_time=None
            
            

    if status and not detection:
             initial_time = time.time()
            
            
    elif detection:
        if timer_started:
            if time.time() - initial_time >= patience:
                detection = False
                status = False
                timer_started = False
                
                print('Stop Recording!')
                initial_time=None
                
                if(name=='Unknown'):
                 count+=1
                 files = glob.glob("C:\\Users\\Admin\\Desktop\\IDS demo\\intruders\\*.jpg")
                 for file in files:
                  with open(file, 'rb') as f:
                   image_data = f.read()
                   image_type = imghdr.what(f.name)
                   image_name = f.name
                   newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)
              
                if(name=='Unknown'):
                 count=0
                 with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    
                   smtp.login(Sender_Email, Password)              
                   smtp.send_message(newMessage)

                   print("Done sending the email.")
                   out.release()
                   
                   body = "Alert: \n A Person Entered the Room at {} \n Left the room at {}".format(entry_time, exit_time)
                   send_message(body, info_dict)
               
        else:
            timer_started = True
            initial_time = time.time()
           
    
    if initial_time is None:
           
        text = 'Patience: {}'.format(patience)
    else: 
        text = 'Patience: {:.2f}'.format(max(0, patience - (time.time() - initial_time)))
       
    cv2.putText(frame, text, (10, 450), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 40, 155) , 2) 
    
    if detection:
      
        out.write(frame)  
             
    face_locations = face_recognition.face_locations(gray)
    face_encodings = face_recognition.face_encodings(gray, face_locations)
    for (x, y, width, height), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(image_encodings, face_encoding)
        name = 'Unknown'
        face_distances = face_recognition.face_distance(image_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = image_names[best_match_index]
        if(name=='Unknown'):
            cv2.imwrite('C:\\Users\\Admin\\Desktop\\IDS demo\\intruders\\intru-{}.jpg'.format(count),frame)
            count+=1
        cv2.rectangle(frame, (height, x), (y, width), (0, 0, 255), 3)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (height + 6, width - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow("output",frame)
    if(cv2.waitKey(1)==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
