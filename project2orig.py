import cv2
import numpy as np
import random

# helper functions
# # get Euclidean distance of 2 vectors
# def euclidean_distance(vec1, vec2):
#     return np.sqrt(np.sum((vec1 - vec2) ** 2))

# get Euclidean distance of one descriptor to all other descriptors
def euclidean_dist(descriptors, ref_descriptor):
    return np.sqrt(np.sum((descriptors - ref_descriptor) ** 2, axis=1)) # axis=1 to sum across the array descriptor as opposed to axis=0 which sums down

# check if 3 points are basically collinear
def check_collinear(p1, p2, p3, thresh=0.001):
    area = abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0)
    return area < thresh

# stage 1- load images 
# load test and reference images
def load_images():
    ref_img = cv2.imread('reference.png')
    filename = input().rstrip()
    test_img = cv2.imread(filename, cv2.IMREAD_COLOR)

    if ref_img is None: # error handling 
        raise FileNotFoundError("Reference image not found.")
    if test_img is None:
        raise FileNotFoundError(f"Test image: {filename} not found.")
    return ref_img, test_img

# stage 2- extract keypoints 
# use sift to get all keypoints and descriptors
def sift_get_keypts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
    sift = cv2.SIFT_create()  # create sift object to perform sift detection
    keypt, des = sift.detectAndCompute(gray, None) # get list of keypoints and corresponding descriptor array Nx128
    return keypt, des

# stage 3- filter best keypoints 
# keep the top 3000 keypoints by response for both images
def filter_top_keypoints(kp_ref, des_ref, kp_test, des_test, max_kp=3000):
    responses_ref = np.array([keypt.response for keypt in kp_ref]) # get the strength of each keypoint and store in arrays using the .response operator
    responses_test = np.array([keypt.response for keypt in kp_test])

    ref_idx = np.argsort(responses_ref)[::-1][:max_kp] # get indices that would sort the responses in ascending order, then reverse to descending order
    test_idx = np.argsort(responses_test)[::-1][:max_kp]

    # return filtered keypoints and descriptors for both images
    return ([kp_ref[i] for i in ref_idx], des_ref[ref_idx], [kp_test[i] for i in test_idx], des_test[test_idx])

# stage 4- keypoint matching
# match descriptors using Lowe’s ratio test 
def match_kps(des_ref_filtered, des_test_filtered, thresh=0.7):
    matches = []
    for i, d_ref in enumerate(des_ref_filtered): # get Euclidean distances between one reference descriptor and all test descriptors
        distances = euclidean_dist(des_test_filtered, d_ref)
        if len(distances) < 2: 
            continue# skip if there are not enough descriptors to compare
        idx_sorted = np.argsort(distances) # get indices of the 2 closest matches
        # best match must be significantly better than the second-best Lowe's ratio
        if distances[idx_sorted[0]] / distances[idx_sorted[1]] < thresh:
            matches.append((i, idx_sorted[0])) # add matches that passed Lowe's ratio test

    return [m for m in matches] # return most consistent matches that passed ratio test

# stage 5- RANSAC
# affine estimate using ransac
def ransac_affine(kp_ref, kp_test, matches, num_iters=3000, thresh=5.0):
    src_pts = np.array([kp_ref[i].pt for i, _ in matches], dtype=np.float32)   # extract matched keypoint coordinates as arrays
    dst_pts = np.array([kp_test[j].pt for _, j in matches], dtype=np.float32)

    best_inliers = None # variables to track best affine matrix and inliers
    best_inlier_count = 0
    best_M = None

    for a in range(num_iters):  # RANSAC loop to find most consistent model
        idx = random.sample(range(len(matches)), 3) # randomly sample 3 matches
        src_sample, dst_sample = src_pts[idx], dst_pts[idx]

        if check_collinear(src_sample[0], src_sample[1], src_sample[2]): # check for collinear points
            continue

        try: # compute affine transformation for this random sample
            M = cv2.getAffineTransform(src_sample, dst_sample)
        except:
            continue # skip invalid transformations

        # check if transformation is degenerate by computing determinant of the linear part
        a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        det = abs(a*d - b*c)
        if det < 0.001 or det > 1000:  # ensure det is not too small and too large
            continue
        
        # project all source points onto test image using M
        projected = np.array([
            [M[0,0]*x + M[0,1]*y + M[0,2], M[1,0]*x + M[1,1]*y + M[1,2]]
            for x, y in src_pts
        ])

        
        # compute Euclidean distances between projected points and actual destination points
        distances = np.sqrt(np.sum((projected - dst_pts)**2, axis=1))
        inliers = distances < thresh   # identify inlier points which have a projection error below the threshold
        inlier_count = np.sum(inliers)

        if inlier_count > best_inlier_count: # update best affine matrix if this model has more inliers
            best_inlier_count = inlier_count
            best_M = M
            best_inliers = inliers

    if best_M is None or best_inlier_count < 3: # if no valid model found return None
        return None, None

    if best_inlier_count > 3: # re-fit the affine transform using all inliers
        best_M = fit_affine_least_squares(src_pts[best_inliers], dst_pts[best_inliers])

    return best_M, best_inliers # return best affine transformation matrix and the inlier mask

# stage 6: refine affine matrix using least squares fitting
def fit_affine_least_squares(src_pts, dst_pts):
    N = src_pts.shape[0]  # num of pt matches

    # initialize least squares formula: A_mat * affine coeefficient = b_vec
    A_mat = np.zeros((2 * N, 6)) # A_mat has 2N rows (x and y equations per point) and 6 unknowns (affine parameters)
    b_vec = np.zeros((2 * N,))

    # build A_mat and b_vec using the point correspondences
    for i in range(N):
        x, y = src_pts[i] # source point coordinates
        x_p, y_p = dst_pts[i] # corresponding destination point coordinates
        
        # fill x-equation row: a*x + b*y + tx = x'
        A_mat[2 * i, :3] = [x, y, 1]
        b_vec[2 * i] = x_p

        # fill y-equation row: c*x + d*y + ty = y'
        A_mat[2 * i + 1, 3:] = [x, y, 1]
        b_vec[2 * i + 1] = y_p

    # solve for affine parameters using least squares to minimize total squared error between projected and actual points
    aff_values, *_ = np.linalg.lstsq(A_mat, b_vec)

    # reshape into a 2x3 affine transformation matrix
    M = np.array([
        [aff_values[0], aff_values[1], aff_values[2]],  # first row: [a, b, tx]
        [aff_values[3], aff_values[4], aff_values[5]]   # second row: [c, d, ty]
    ])
    return M  # return refined affine transformation matrix

# stage 7- bounding box 
# build bounding box
def compute_bounding_box(ref_img, M):
    ref_h, ref_w = ref_img.shape[:2] # get reference image dimensions
    ref_corners = np.array([[0,0],[ref_w,0],[ref_w,ref_h],[0,ref_h]]) # transform reference bounding box corners to test image

    # apply affine transformation to each corner point
    test_corners = np.array([
        [M[0,0]*x + M[0,1]*y + M[0,2], M[1,0]*x + M[1,1]*y + M[1,2]]
        for x, y in ref_corners
    ])

    # find center and dimensions from transformed corners
    center = np.mean(test_corners, axis=0) # compute avg of the 4 transformed corner coordinates
    X, Y = int(round(center[0])), int(round(center[1])) #  center of bounding box in test image

    scale_x = np.sqrt(M[0,0]**2 + M[1,0]**2) # get bounding box height/width from the scale of the transformation using scale of the transformation matrix
    scale_y = np.sqrt(M[0,1]**2 + M[1,1]**2)
    H = round(ref_h * (scale_x + scale_y) / 2) # height is the scaled reference height

    A = round(np.degrees(np.arctan2(M[1,0], M[0,0])) % 360) # get angle from transformation matrix M
    X = max(0, min(X, 1999))
    Y = max(0, min(Y, 1499))
    H = max(1, min(H, 1499))

    return X, Y, H, A # display final output

# main function
def main():
    ref_img, test_img = load_images() # load images
    kp_ref, des_ref = sift_get_keypts(ref_img) # get keypoints and descriptors for reference image using sift
    kp_test, des_test = sift_get_keypts(test_img) # get keypoints and descriptors for test image using sift

    kp_ref_filtered, des_ref_filtered, kp_test_filtered, des_test_filtered = filter_top_keypoints(kp_ref, des_ref, kp_test, des_test, max_kp=3000) # keep only the best keypoints

    matches = match_kps(des_ref_filtered, des_test_filtered) # match descriptors using lowe's ratio
    if len(matches) < 3:
        print("0 0 0 0")
        return

    M, inliers = ransac_affine(kp_ref_filtered, kp_test_filtered, matches) # get best affine transformation matrix and the inlier mask
    if M is None:
        print("0 0 0 0")
        return

    X, Y, H, A = compute_bounding_box(ref_img, M) # print bounding box coordinates
    print(f"{X} {Y} {H} {A}")

if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# Project 2 Computer Vision (CAI4841)
# Joshua Maharaj U24183946
# Acknowledgment:
# Portions of this code such as the for loop from lines 131-141 in the 
# fit_affine_least_squares function and the
# code to get the test corners from each point in the test image in lines 155-161
# were refined and modified using ChatGPt after debugging. 
# ------------------------------------------------------------
