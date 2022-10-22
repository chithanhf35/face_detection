import cv2
import imutils
"""camera_id = 0
# Mở camera
cap = cv2.VideoCapture(camera_id)
# Đọc ảnh từ camera
while(True):
    ret, frame = cap.read()
    cv2.imshow("cam",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
# Giải phóng camera
cap.release()"""
img= cv2.imread("b.jpg")
cv2.imshow("xe tank", img)
cv2.waitKey()
img_r= imutils.rotate(img,90)
cv2.imshow("rotate",img_r)
cv2.waitKey()
cv2.destroyAllWindows()
"""# Đặt kích thước cho ảnh
img_rs = cv2.resize(img,dsize=(1000,600))
cv2.imshow("resize",img_rs)
cv2.waitKey()
"""
