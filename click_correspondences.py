'''
  File name: click_correspondences.py
  Author: Tong Pow
  Date created: 10/5/17
'''
from morphing.cpselect import cpselect, cpselect_recorder
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tempfile import TemporaryFile

'''
  File clarification:
    Click correspondences between two images
    - Input im1: target image
    - Input im2: source image
    - Output im1_pts: correspondences coordiantes in the target image
    - Output im2_pts: correspondences coordiantes in the source image
'''

def click_correspondences(im1, im2):
  '''
    Tips:
      - use 'matplotlib.pyplot.subplot' to create a figure that shows the source and target image together
      - add arguments in the 'imshow' function for better image view
      - use function 'ginput' and click correspondences in two images in turn
      - please check the 'ginput' function documentation carefully
        + determine the number of correspondences by yourself which is the argument of 'ginput' function
        + when using ginput, left click represents selection, right click represents removing the last click
        + click points in two images in turn and once you finish it, the function is supposed to 
          return a NumPy array contains correspondences position in two images
  '''
  #Create figure, show source and target image together
  fig = plt.figure()
  a = fig.add_subplot(1, 2, 1)

  img1 = mpimg.imread(im1)
  lum_img1 = img1[:,:,:]
  imgplot = plt.imshow(lum_img1, cmap='hot')
  imgplot.set_clim(0.0, 0.7)
  a.set_title('Before')
  plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
  a = fig.add_subplot(1, 2, 2)

  img2 = mpimg.imread(im2)
  lum_img2 = img2[:,:,:]
  imgplot = plt.imshow(lum_img2, cmap='hot')
  imgplot.set_clim(0.0, 0.7)
  a.set_title('After')
  plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
  #plt.show()
  print("lum_img1 shape: ", img1.shape)
  print("lum_img2 shape: ", img2.shape)

  point1, point2 = cpselect(img1, img2)
  print("point1 = ", point1)
  print("point2 = ", point2)



  return point1, point2

im1 = 'Tong Portrait 2c.jpg'
im2 = 'Gosling.jpg'
im1_matrix = np.asarray(im1)

#[im1_pts, im2_pts] = click_correspondences(im1, im2)