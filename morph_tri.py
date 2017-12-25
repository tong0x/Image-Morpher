'''
  File name: morph_tri.py
  Author:
  Date created:
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.linalg import inv
from PIL import Image as im
import matplotlib.image as mpimg
from scipy.spatial import Delaunay
import imageio


def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):

  ones = np.ones(len(warp_frac))

  # Compute barycentric coordinates for every single point of intermediate image
  ys, xs, colors = im1.shape

  # Declare new intermediate image matrix
  im_halfway_matrix = np.zeros((len(warp_frac), ys, xs, 3))

  for k in range(0, len(warp_frac)):
    # Compute intermediate shape based on warp fraction
    im_halfway = (1 - warp_frac[k]) * im1_pts + warp_frac[k] * im2_pts

    # Generate Delaunay triangulation from intermediate control points
    tri = Delaunay(im_halfway)

    # Apply Delaunay triangulation to source and target images
    tri_source = Delaunay(im1_pts)
    tri_target = Delaunay(im2_pts)

    # Array of indexes of triangle corners
    simplices = tri.simplices
    print("Simplices!!", simplices)
    simplices_target = tri_target.simplices
    simplices_source = tri_source.simplices
    for j in range(0, ys):
      for i in range(0, xs):
        # find_simplex returns the triangle number
        simplex = tri.find_simplex(np.array([i, j]))
        corner_indices = simplices[simplex]
        # Get exact coordinates of corner points and append 1 to make columns of matrix A
        point1 = np.array([im_halfway[corner_indices[0]][0], im_halfway[corner_indices[0]][1], 1])
        point2 = np.array([im_halfway[corner_indices[1]][0], im_halfway[corner_indices[1]][1], 1])
        point3 = np.array([im_halfway[corner_indices[2]][0], im_halfway[corner_indices[2]][1], 1])
        # Combine into matrix A for intermediate image
        A = np.column_stack((point1, point2, point3))
        # Get barycentric coordinates in the form of a vector
        coordinate_vector = np.array([[i], [j], [1]])
        curr_barycentric_vector = np.dot(inv(A), coordinate_vector)

        # Get matrix A for target image
        corner_indices_target = simplices[simplex]
        point1_target = np.array([im2_pts[corner_indices_target[0]][0], im2_pts[corner_indices_target[0]][1], 1])
        point2_target = np.array([im2_pts[corner_indices_target[1]][0], im2_pts[corner_indices_target[1]][1], 1])
        point3_target = np.array([im2_pts[corner_indices_target[2]][0], im2_pts[corner_indices_target[2]][1], 1])
        A_target = np.column_stack((point1_target, point2_target, point3_target))

        # Get target image's coordinates, round them to integers
        coordinate_target = np.dot(A_target, curr_barycentric_vector)
        x_target = int(np.round(coordinate_target[0] / coordinate_target[2])[0])
        y_target = int(np.round(coordinate_target[1] / coordinate_target[2])[0])

        # Get matrix A for source image
        corner_indices_source = simplices[simplex]
        point1_source = np.array([im1_pts[corner_indices_source[0]][0], im1_pts[corner_indices_source[0]][1], 1])
        point2_source = np.array([im1_pts[corner_indices_source[1]][0], im1_pts[corner_indices_source[1]][1], 1])
        point3_source = np.array([im1_pts[corner_indices_source[2]][0], im1_pts[corner_indices_source[2]][1], 1])
        A_source = np.column_stack((point1_source, point2_source, point3_source))
        # Get source image's coordinates, round them to integers
        coordinate_source = np.dot(A_source, curr_barycentric_vector)
        x_source = int(np.round(coordinate_source[0] / coordinate_source[2])[0])
        y_source = int(np.round(coordinate_source[1] / coordinate_source[2])[0])

        # Cross dissolve Remember 4th color dimension
        im_halfway_matrix[k, i, j, :] = np.round((1 - dissolve_frac[k]) * im1[x_source, y_source, :] +
                                                 dissolve_frac[k] * im2[x_target, y_target, :])

  return im_halfway_matrix.astype('uint8')


# Add photo names here to morph from im1 to im2
im1 = 'Tong Portrait 2c.jpg'
im2 = 'Gosling.jpg'
img1 = mpimg.imread(im1)
img2 = mpimg.imread(im2)

# Morph points: can also use click_correspondences function to manually select them
im1_pts = np.array([[0, 0], [0, 499], [499, 0], [499, 499], [204, 248],
                    [312, 254],
                    [250, 315],
                    [240, 436],
                    [128, 275],
                    [381, 294],
                    [147, 115],
                    [386, 132],
                    [148, 425],
                    [356, 451],
                    [268, 176],
                    [151, 367],
                    [342, 392],
                    [275, 41],
                    [261, 226],
                    [264, 130]])
im2_pts = np.array([[0, 0], [0, 499], [499, 0], [499, 499], [195, 260],
                    [295, 262],
                    [249, 334],
                    [256, 469],
                    [118, 315],
                    [374, 293],
                    [108, 147],
                    [378, 139],
                    [132, 479],
                    [401, 467],
                    [235, 174],
                    [148, 389],
                    [357, 370],
                    [223, 49],
                    [240, 230],
                    [225, 125]])
warp_frac = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
dissolve_frac = warp_frac
morph_im = morph_tri(img1, img2, im1_pts, im2_pts, warp_frac, dissolve_frac)
height, width, color = img1.shape

num_photos, x, y, z = morph_im.shape
morph_list = []
for i in range(0, num_photos):
  image = im.fromarray(morph_im[i, :, :, :], 'RGB')
  morph_list.append(morph_im[i, :, :, :])
  image.save('photo.jpg')
  image.show()
  plt.show()
  imageio.mimsave('./eval_morphimg2.gif', morph_list)
