import cv2
from osgeo import gdal
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
import argparse

parser = argparse.ArgumentParser()

#-db DATABSE -u USERNAME -p PASSWORD -size 20
parser.add_argument("-s", "--source", help="Source Filename - to be matched to Target")
parser.add_argument("-t", "--target", help="Target Filename")

args = parser.parse_args()
print("SOUCE: ", args.source)
print("TARGET: ", args.target)

# Importing the images to be matched. SRC is the image to be mapped/corrected to TARGET
src_filename = str(args.source)
target_filename = str(args.target)

# Opening the files and importing the imaging bands as layers
src = gdal.Open(src_filename)
print ("band count: " + str(src.RasterCount))
src1_orig = np.array(src.GetRasterBand(1).ReadAsArray())
src2_orig = np.array(src.GetRasterBand(2).ReadAsArray())
src3_orig = np.array(src.GetRasterBand(3).ReadAsArray())

del src # Closing the database

target = gdal.Open(target_filename)
print ("band count: " + str(target.RasterCount))
target1 = np.array(target.GetRasterBand(1).ReadAsArray())
target2 = np.array(target.GetRasterBand(2).ReadAsArray())
target3 = np.array(target.GetRasterBand(3).ReadAsArray())

del target # Closing the database


# Normalizing to UINT8
src1 = cv2.normalize(src1_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
src2 = cv2.normalize(src2_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
src3 = cv2.normalize(src3_orig, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

target1 = cv2.normalize(target1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
target2 = cv2.normalize(target2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
target3 = cv2.normalize(target3, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


# Stacking the satellite image bands into a single image for both SRC and TARGET
norm_src_rgb = np.dstack((src1,src2, src3))
norm_target_rgb = np.dstack((target1,target2, target3))

# Normalizing and converting the images to properly handle them as color images for CV2
norm_src_gray = cv2.cvtColor(norm_src_rgb, cv2.COLOR_BGR2GRAY)
norm_target_gray = cv2.cvtColor(norm_target_rgb, cv2.COLOR_BGR2GRAY)

# Begin processing with CV2. Involves histogram equalization and SIFT feature detection and matching
img1 = norm_src_gray
img2 = norm_target_gray

# Matching the historgrams between source and target. This can imporove the number of matches for SIFT
img1 = exposure.match_histograms(img1, img2)
img1 = img1.astype(np.uint8)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.70*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,0,255), # draw matches in red color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#plt.imshow(img3, 'gray'),plt.show()
cv2.imwrite("matched.png", img3)

# Removing the scaling factor from the homography matrix
M[0,0] = 1.0
M[1,1] = 1.0

# Running some processing on the matching points to further select the best points
bool_list = list(map(bool,matchesMask))
src_points = src_pts[bool_list]
dst_points = dst_pts[bool_list]
sp = np.asarray(src_points)
dp = np.asarray(dst_points)

sp_y = sp[:, 0, 0]
sp_x = sp[:, 0, 1]
dp_y = dp[:, 0, 0]
dp_x = dp[:, 0, 1]

# height, width of both source and target
height_sp = img1.shape[0]
width_sp = img1.shape[1]
height_dp = img2.shape[0]
width_dp = img2.shape[1]

# Normalizing to get the amount of pixel shift needed for the gdal geotransform
sp_y_norm = sp_y/height_sp
sp_x_norm = sp_y/width_sp
dp_y_norm = dp_y/height_dp
dp_x_norm = dp_y/width_dp

x_shift_percent = np.subtract(sp_x_norm, dp_x_norm)
y_shift_percent = np.subtract(sp_y_norm, dp_y_norm)

x_shift_percent = list(filter(lambda num: ((num <= 0.01) and (num >= -0.01)) , x_shift_percent))
y_shift_percent = list(filter(lambda num: ((num <= 0.01) and (num >= -0.01)) , y_shift_percent))

x_pixel_shift = np.median(np.array(x_shift_percent))
y_pixel_shift = np.median(np.array(y_shift_percent))
x_shift_percent_mean = x_pixel_shift
y_shift_percent_mean = y_pixel_shift 
x_shift_std = np.subtract(sp_x_norm, dp_x_norm).std()
y_shift_std = np.subtract(sp_y_norm, dp_y_norm).std()

# Applying the Homography Transform to the Bands of the SRC image
rows, cols = img1.shape
src1_warp = cv2.warpPerspective(src1_orig, M, (cols, rows))
src2_warp = cv2.warpPerspective(src2_orig, M, (cols, rows))
src3_warp = cv2.warpPerspective(src3_orig, M, (cols, rows))

# Writing the warped image bands to a new file with the same parameters as the source file
import rasterio
with rasterio.open(src_filename) as src_dataset:

    # Get a copy of the source dataset's profile. Thus our
    # destination dataset will have the same dimensions,
    # number of bands, data type, and georeferencing as the
    # source dataset.
    kwds = src_dataset.profile

    # Change the format driver for the destination dataset to
    # 'GTiff', short for GeoTIFF.
    kwds['driver'] = 'GTiff'

    with rasterio.open('src_warped.tif', 'w', **kwds) as dst_dataset:
        dst_dataset.write(src1_warp, 1)
        dst_dataset.write(src2_warp, 2)
        dst_dataset.write(src3_warp, 3)


# open dataset with update permission to update the coordinates to reflect the pixel shift/translation
ds = gdal.Open('src_warped.tif', gdal.GA_Update)
# get the geotransform as a tuple of 6
gt = ds.GetGeoTransform()
# unpack geotransform into variables
x_tl, x_res, dx_dy, y_tl, dy_dx, y_res = gt

# compute shift of N pixel(s) in X direction
shift_x = -x_pixel_shift * x_res
# compute shift of M pixels in Y direction
# y_res likely negative, because Y decreases with increasing Y index
shift_y = (-y_pixel_shift/2)  * y_res

# make new geotransform
gt_update = (x_tl + shift_x, x_res, dx_dy, y_tl + shift_y, dy_dx, y_res)
# assign new geotransform to raster
ds.SetGeoTransform(gt_update)
# ensure changes are committed
ds.FlushCache()
ds = None