from importlib.resources import path
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print('warring warring')
                        t1=threading.Thread(target=playsound(r'C:\Users\chith\OneDrive\Desktop\2022_Python\alert.mp3'), args=('wake up',))
                        t1.start()
                        t1.join()
                        cv2.putText(frame,"DROWSINESS",(10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.drawContours(frame, [leftEyeHull], -1,  (0, 255, 0),1)
                        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0),1)