#!/usr/bin/env python3
"""
project2.py

Usage:
    Provide the test image filename on stdin (the grader will do this).
    Example (during development): echo "test.png" | python3 project2.py

Notes:
- Uses cv2 and numpy only.
- Uses OpenCV to detect and compute keypoints/descriptors (SIFT).
- Matching (nearest neighbor + ratio test) is implemented manually (no cv2 matchers).
- Uses affine estimation from triplets (cv2.getAffineTransform) inside a simple RANSAC loop.
- Computes oriented bounding box center (X,Y), height (H), and angle (A) in degrees clockwise.
- If matching fails, outputs a safe fallback: centered with H=0 and A=0.

"""

import sys
import cv2
import numpy as np
import math
import time

# --- PARAMETERS (tweak for robustness / speed) ---
MAX_REF_KP = 2000           # limit reference keypoints (SIFT) for speed
MAX_TEST_KP = 2000          # limit test keypoints
NN_RATIO = 0.75             # Lowe ratio test threshold
RANSAC_ITERS = 1500         # RANSAC iterations for triplet sampling
INLIER_DIST = 10.0          # pixels: distance threshold to consider a match an inlier
MIN_INLIERS = 12            # minimum inliers to accept a model
TIME_LIMIT = 18.0           # seconds (safe margin under 20s)

# -------------------------------------------------

def read_images(test_filename):
    ref = cv2.imread('reference.png', cv2.IMREAD_COLOR)
    test = cv2.imread(test_filename, cv2.IMREAD_COLOR)
    if ref is None:
        raise FileNotFoundError("reference.png not found in current directory")
    if test is None:
        raise FileNotFoundError(f"test image '{test_filename}' not found or cannot be read")
    return ref, test

def detect_and_describe(img, use_sift=True, max_kp=2000):
    """Detect keypoints and compute descriptors. Returns keypoints (list) and descriptors (numpy array)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if use_sift:
        # SIFT is robust; fallback to ORB if not available
        try:
            sift = cv2.SIFT_create(nfeatures=max_kp)
        except AttributeError:
            # older opencv versions or unavailable -> use ORB
            orb = cv2.ORB_create(nfeatures=max_kp)
            kp, des = orb.detectAndCompute(gray, None)
            return kp, des
        kp, des = sift.detectAndCompute(gray, None)
        # SIFT returns float32 descriptors
        return kp[:max_kp], (des[:max_kp].astype(np.float32) if des is not None else None)
    else:
        orb = cv2.ORB_create(nfeatures=max_kp)
        kp, des = orb.detectAndCompute(gray, None)
        return kp[:max_kp], des

def descriptors_to_array(des):
    """Ensure descriptors are a numpy array (None safe)."""
    if des is None:
        return None
    return np.asarray(des)

def match_descriptors_manual(desc1, desc2, is_float=True, ratio=0.75):
    """
    Manual nearest neighbor matching with Lowe ratio test.
    desc1: NxD descriptors (reference)
    desc2: MxD descriptors (test)
    returns list of matches: [(i, j, dist_ij), ...] where i indexes desc1 and j indexes desc2
    """
    if desc1 is None or desc2 is None:
        return []

    # For ORB (uint8 binary), we can compute Hamming via bitwise xor -> count non-zero bits,
    # but using numpy broadcasting with bit_count is not as trivial. We will convert to float and use Euclidean,
    # which is acceptable and simpler; for binary descriptors this is still okay.
    # To keep it efficient, we compute in chunks if needed.
    A = desc1.astype(np.float32)
    B = desc2.astype(np.float32)

    matches = []

    # To avoid building the full NxM matrix if very large, process in chunks of ref descriptors
    CHUNK = 512
    N = A.shape[0]
    for start in range(0, N, CHUNK):
        end = min(N, start + CHUNK)
        a_chunk = A[start:end]  # shape (k, D)
        # Compute squared distances: (a-b)^2 = a^2 + b^2 - 2ab
        aa = np.sum(a_chunk * a_chunk, axis=1)[:, None]  # (k,1)
        bb = np.sum(B * B, axis=1)[None, :]             # (1,M)
        ab = np.dot(a_chunk, B.T)                       # (k,M)
        d2 = aa + bb - 2.0 * ab                         # (k, M)
        # Numerical issues
        d2 = np.maximum(d2, 0.0)
        # find best and second best
        idx1 = np.argmin(d2, axis=1)
        best = d2[np.arange(d2.shape[0]), idx1]
        # mask them out to find second best
        d2_masked = d2.copy()
        d2_masked[np.arange(d2.shape[0]), idx1] = np.inf
        idx2 = np.argmin(d2_masked, axis=1)
        second = d2_masked[np.arange(d2.shape[0]), idx2]

        for k, (i2, b2, s2) in enumerate(zip(idx1, best, second)):
            # ratio test on sqrt distances (Euclidean)
            if s2 == np.inf:
                continue
            if math.sqrt(b2) < ratio * math.sqrt(s2):
                matches.append((start + k, int(i2), float(math.sqrt(b2))))
    return matches

def ransac_affine(ref_pts, test_pts, matches, img_shape, time_budget=TIME_LIMIT):
    """
    RANSAC over affine transformations computed from 3-point correspondences.
    ref_pts: Nx2 numpy array of reference keypoint coordinates
    test_pts: Mx2 numpy array of test keypoint coordinates
    matches: list of (i_ref, j_test, dist)
    Returns best affine 2x3 matrix and mask of inliers (indices into matches list)
    """
    if len(matches) < 3:
        return None, []

    # Precompute matched point arrays for speed
    ref_idx = np.array([m[0] for m in matches], dtype=int)
    test_idx = np.array([m[1] for m in matches], dtype=int)
    src = ref_pts[ref_idx]   # Nx2
    dst = test_pts[test_idx] # Nx2

    best_M = None
    best_inliers = []
    best_count = 0
    n_matches = src.shape[0]

    rng = np.random.default_rng(seed=12345)
    t_start = time.time()

    # RANSAC loop
    iters = RANSAC_ITERS
    for it in range(iters):
        # Time check
        if time.time() - t_start > time_budget:
            break

        # randomly pick 3 distinct indices
        ids = rng.choice(n_matches, size=3, replace=False)
        p_src = src[ids].astype(np.float32)
        p_dst = dst[ids].astype(np.float32)

        # cv2.getAffineTransform needs shape (3,2)
        try:
            M = cv2.getAffineTransform(p_src, p_dst)  # 2x3 float32
        except Exception:
            continue

        # apply M to all src points
        ones = np.ones((n_matches, 1), dtype=np.float32)
        src_hom = np.hstack([src.astype(np.float32), ones])  # Nx3
        pred = (M.reshape(2,3) @ src_hom.T).T  # Nx2

        # distances
        dists = np.linalg.norm(pred - dst, axis=1)

        inlier_mask = dists <= INLIER_DIST
        inlier_count = int(np.count_nonzero(inlier_mask))

        # update best
        if inlier_count > best_count and inlier_count >= MIN_INLIERS:
            best_count = inlier_count
            best_inliers = np.nonzero(inlier_mask)[0].tolist()
            best_M = M.copy()

            # early exit if very good
            if best_count > 0.8 * n_matches:
                break

    # If we have a decent model, refine using all inliers via least-squares affine (cv2.estimateAffinePartial2D allowed? but it's OpenCV matching function)
    # Simpler: re-compute M using cv2.getAffineTransform on best 3 inliers or do a least squares fit:
    if best_M is not None and len(best_inliers) >= 3:
        # Fit using a simple least squares: find affine A (2x3) minimizing ||A*[x,y,1]^T - [u,v]^T||
        in_src = src[best_inliers].astype(np.float32)
        in_dst = dst[best_inliers].astype(np.float32)
        # Build matrix for least squares: for each point x,y -> rows for u and v
        # [ x y 1 0 0 0 ] [a11 a12 a13]^T = u
        # [ 0 0 0 x y 1 ] [a21 a22 a23]^T = v
        k = in_src.shape[0]
        A = np.zeros((2*k, 6), dtype=np.float32)
        b = np.zeros((2*k,), dtype=np.float32)
        for i in range(k):
            x, y = in_src[i]
            u, v = in_dst[i]
            A[2*i, 0:3] = [x, y, 1]
            A[2*i+1, 3:6] = [x, y, 1]
            b[2*i] = u
            b[2*i+1] = v
        # Solve least squares
        try:
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            M_refined = np.array([[sol[0], sol[1], sol[2]],
                                   [sol[3], sol[4], sol[5]]], dtype=np.float32)
            best_M = M_refined
        except Exception:
            pass

    return best_M, best_inliers

def compute_bbox_from_affine(M, ref_shape):
    """
    Given affine M (2x3) mapping reference coords -> test coords,
    and the reference image shape (h_ref, w_ref), assume the bounding box of Rocky in
    the reference is the whole reference image (center and full height). Compute (cx, cy, H, angle).
    Returns integers (cx, cy, H, angle)
    """
    h_ref, w_ref = ref_shape[0], ref_shape[1]
    # reference box center and top/bottom center in ref coordinates
    cx_ref = w_ref / 2.0
    cy_ref = h_ref / 2.0
    top_ref = np.array([cx_ref, cy_ref - h_ref/2.0], dtype=np.float32)
    bot_ref = np.array([cx_ref, cy_ref + h_ref/2.0], dtype=np.float32)
    center_ref = np.array([cx_ref, cy_ref], dtype=np.float32)

    def transform_pt(pt):
        x, y = float(pt[0]), float(pt[1])
        u = M[0,0]*x + M[0,1]*y + M[0,2]
        v = M[1,0]*x + M[1,1]*y + M[1,2]
        return np.array([u, v], dtype=np.float32)

    top_t = transform_pt(top_ref)
    bot_t = transform_pt(bot_ref)
    center_t = transform_pt(center_ref)

    # height is distance between top and bottom transformed centers
    v = bot_t - top_t
    height = float(np.linalg.norm(v))

    # angle: clockwise rotation from vertical to v
    vx, vy = float(v[0]), float(v[1])
    # If height is ~0, fallback angle 0
    if height < 1e-6:
        angle = 0.0
    else:
        # atan2(vx, vy) gives angle between vertical (0,1) and v, positive when rotated clockwise
        angle_rad = math.atan2(vx, vy)
        angle = math.degrees(angle_rad)
        # normalize to [0,360)
        angle = angle % 360.0

    cx = int(round(float(center_t[0])))
    cy = int(round(float(center_t[1])))
    H = int(round(height))
    A = int(round(angle)) % 360
    return cx, cy, H, A

def main():
    # read filename from stdin as required by assignment
    filename = sys.stdin.readline().rstrip()
    if filename == '':
        # If interactive testing, allow a filename via command-line argument
        if len(sys.argv) >= 2:
            filename = sys.argv[1]
        else:
            print("0 0 0 0")
            return

    try:
        ref_img, test_img = read_images(filename)
    except FileNotFoundError as e:
        print("0 0 0 0")
        return

    # Detect keypoints and descriptors
    kp_ref, des_ref = detect_and_describe(ref_img, use_sift=True, max_kp=MAX_REF_KP)
    kp_test, des_test = detect_and_describe(test_img, use_sift=True, max_kp=MAX_TEST_KP)

    # Convert to arrays
    des_ref_arr = descriptors_to_array(des_ref)
    des_test_arr = descriptors_to_array(des_test)

    # Build list of keypoint coordinates
    ref_pts = np.array([kp.pt for kp in kp_ref], dtype=np.float32) if kp_ref is not None else np.zeros((0,2), dtype=np.float32)
    test_pts = np.array([kp.pt for kp in kp_test], dtype=np.float32) if kp_test is not None else np.zeros((0,2), dtype=np.float32)

    # Manual matching (nearest neighbor + ratio test)
    matches = match_descriptors_manual(des_ref_arr, des_test_arr, is_float=True, ratio=NN_RATIO)
    # matches is list of (i_ref, j_test, dist)
    if len(matches) == 0:
        # fallback: no matches found
        W = test_img.shape[1]
        H_img = test_img.shape[0]
        # safe fallback: center of image, zero height
        print(f"{W//2} {H_img//2} 0 0")
        return

    # RANSAC to find best affine
    M, inliers = ransac_affine(ref_pts, test_pts, matches, test_img.shape, time_budget=TIME_LIMIT)
    if M is None:
        # Fallback: use the best single match to estimate a small box
        # Use the match with smallest descriptor distance
        matches_sorted = sorted(matches, key=lambda x: x[2])
        best = matches_sorted[0]
        j_test = best[1]
        x, y = test_pts[j_test]
        # small guess: height = 100
        h_guess = 200
        # output center and zero angle
        cx = int(round(x))
        cy = int(round(y))
        H_out = int(h_guess)
        A_out = 0
        print(f"{cx} {cy} {H_out} {A_out}")
        return

    # Compute oriented bounding box parameters from affine
    cx, cy, H_box, A_deg = compute_bbox_from_affine(M, ref_img.shape)
    # Clamp results to image boundaries and sensible ranges
    W_test = test_img.shape[1]
    H_test = test_img.shape[0]
    cx = max(0, min(W_test-1, cx))
    cy = max(0, min(H_test-1, cy))
    H_box = max(0, min(H_test-1, H_box))

    print(f"{cx} {cy} {H_box} {A_deg}")

if __name__ == "__main__":
    main()
