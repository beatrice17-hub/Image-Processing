import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# 1. Memasukkan gambar buah
image_bgr = cv2.imread('apel_putih2.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 2. Membuat Mask dan melakukan Grabcut pada gambar buah
rectangle =(230, 95, 290, 300)
mask = np.zeros(image_rgb.shape[:2],np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(image_rgb, mask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# 3. Menghilangkan background gambar (merubah menjadi hitam (0))
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
image_rgb_nobg1 = cv2.cvtColor(image_rgb_nobg, cv2.COLOR_BGR2RGB)
cv2.imwrite('NoBackGround.png',image_rgb_nobg1)
NoBackground = cv2.imread('NoBackGround.png')

# 4. Merubah gambar tanpa background menjadi binary
img = np.copy(image_rgb_nobg)
Nonzero=np.where((img[:,:,0]!=0) & (img[:,:,1]!=0) & (img[:,:,2]!=0))
img[Nonzero]=(255)
cv2.imwrite('Gambar_binary.png',img)
GambarFinal = cv2.imread('Gambar_binary.png',0)

# 5. Menghitung luas dari jumlah pixel warna putih (255)
Luas = cv2.countNonZero(GambarFinal)

# 6. Melakukan Erosi untuk menghilangkan batang apel
kernel = np.ones((5,5),np.uint8)
erosi = cv2.erode(GambarFinal,kernel,iterations = 4)

# 7. Melakukan Dilasi untuk mengembalikan bagian gambar buah yang hilang akibat Erosi
dilasi = cv2.dilate(erosi,kernel,iterations = 4)

# 8. Mengisi bagian gambar buah yang bolong dengan floodFill
im_floodfill = dilasi.copy()
h, w = dilasi.shape[:2]
mask1 = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im_floodfill, mask1, (0,0), 255);
im_floodfill_inv = cv2.bitwise_not(im_floodfill) 
im_out = dilasi | im_floodfill_inv

# 9. Mecari keliling menggunakan Canny
Canny = cv2.Canny(im_out,50,200)
Keliling = cv2.countNonZero(Canny)

# 10. Mencari nilai lebar dan tinggi gambar buah
contours, hierarchy = cv2.findContours(im_out,1,2)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
img1 = cv2.rectangle(im_out,(x,y),(w,h),(0,255,0),2)
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
im = cv2.drawContours(Canny,[box],-1,(255,255,255),1)

# 11. Mengatur display dari hasil mengolah gambar buah
titles = ['1.Original Image','2.Mask','3.Image Without Background','4.Binary','5.Erosi','6.Dilasi','7.FloodFill','8.Canny (Zoom Gambar agar tidak putus-putus)']
images = [image_rgb, mask, image_rgb_nobg, GambarFinal, erosi, dilasi, im_out, Canny]
for i in range(8):
        plt.subplot(2,4,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([]),plt.yticks([])
        
# 12. Menghitung nilai rata-rata Red, Green, dan Blue pada buah
class PixelCounter(object):
  ''' loop through each pixel and average rgb '''
  def __init__(self, ITEM):
      self.pic = Image.open(ITEM)
      # load image data
      self.imgData = self.pic.load()
  def averagePixels(self):
      r, g, b = 0, 0, 0
      count = Luas
      for x in range(self.pic.size[0]):
          for y in range(self.pic.size[1]):
              tempr,tempg,tempb = self.imgData[x,y]
              r += tempr
              g += tempg
              b += tempb
      return (r/count), (g/count), (b/count), count
if __name__ == '__main__':
  pc = PixelCounter('NoBackGround.png')
  
# 13. Mencetak nilai-nilai hasil mengolah gambar buah
print ('Lebar:',w,'pixels, ', 'Tinggi:', h, 'pixels')
print ('Luas:', Luas,'pixels')
print ('Keliling:',Keliling,'pixels')
print ('Red','Green','Blue','Total pixel count',pc.averagePixels())
print ('" Maximize windows dan jika gambar canny terlihat putus - putus silahkan di zoom"');
       
plt.show()
