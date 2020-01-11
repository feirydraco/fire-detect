from predict import predict_image
import smtplib 
from skimage import io

#Sender details
email = ""
password = ""

#Recievers email
recv = ""
    
# #Read image
im = io.imread("./inference/fire3.jpeg")

#set up smtp for mail
s = smtplib.SMTP('smtp.gmail.com', 587) 
s.starttls() 
#login
s.login(email, password) 

#Output predictions 
if predict_image(im):
    # 0: fire 1:not-fire
    message = "Not Fire!"
else:
    message = "Fire!"

print(message)
s.sendmail(email, recv, message) 
s.quit()  
