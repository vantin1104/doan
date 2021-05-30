
import cv2 # Khai báo thư viện cv2
import numpy as np # Khai báo thư viện numpy
from matplotlib import pyplot as plt  # Khai báo thư viện matplotlib

img=cv2.imread('lungcancer.jpg') #đọc ảnh ung thư phổi 
img1= cv2.imread('lungcancer.jpg') # đọc ảnh ung thư phổi
img2=cv2.imread('normallung.jpg')  # đọc ảnh phổi bình thường 

med = cv2.medianBlur(img, 5) # Khử nhiễu bằng bộ lọc trung vị ảnh ung thư làm nhiễm muối tiêu ở mức 5
cv2.imwrite('anhkhunhieu.jpg',med)


gray = cv2.cvtColor(med,cv2.COLOR_BGR2GRAY) #Chuyển ảnh ung thư sang ảnh xám
cv2.imwrite('anhxam.jpg',gray)


ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # Dùng hàm phân ngưỡng nhằm đưa ảnh về dạng nhị phân là threshold ở giá trị 127 
cv2.imwrite('phannguong.jpg', gray)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2) # Gán cho biến thresh bằng giá trị lựa chọn ngưỡng động trong vùng lân cận 
cv2.imwrite('phannguongdong.jpg',gray)
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2] #Tìm các đối tượng màu trắng từ nền đen để vẽ  các đường viền contour để phát hiện các vật thể 

for contour in contours:
    cv2.drawContours(gray, contour, -1, (0, 0, 128), 5) # Vẽ contour quanh đối tượng với màu xanh đậm có giá trị (0,0,128)


# Xuất ảnh canh chỉnh với vị trí mong muốn 
plt.subplot(1,3,1),plt.imshow(img2)
plt.title('Anh phoi binh thuong'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(img1)
plt.title('Anh goc'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(gray)
plt.title('Anh phoi ung thu'), plt.xticks([]), plt.yticks([])

plt.show()


