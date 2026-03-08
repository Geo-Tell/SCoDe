import os
import numpy as np
import cv2
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path
from collections import defaultdict
import sys
import time
from scipy.signal import savgol_filter  # Add Savitzky-Golay filter
import pickle  # Add this import at the top with other imports

# Add necessary paths
sys.path.append('/data/nfs/lhj/OverlapEstimation/SCoDe')
sys.path.append('/data/nfs/lhj/OverlapEstimation/SCoDe/third_party/disk')

# Import SCoDe
from dloc.core.overlaps.scode import CCOE as SCoDe

# Import extractors and matchers as available
try:
    from disk.model.disk import DISK
    DISK_AVAILABLE = True
except ImportError:
    print("DISK not available. Only evaluating other methods.")
    DISK_AVAILABLE = False

# Import from dloc.core for consistency with project structure
try:
    from dloc.core.extractors.r2d2 import R2D2Desc
    R2D2_AVAILABLE = True
except ImportError:
    print("R2D2 not available. Skipping this method.")
    R2D2_AVAILABLE = False

try:
    from dloc.core.extractors.d2net import D2Net
    D2NET_AVAILABLE = True
except ImportError:
    print("D2-Net not available. Skipping this method.")
    D2NET_AVAILABLE = False

try:
    from dloc.core.extractors.superpoint import SuperPoint
    SUPERPOINT_AVAILABLE = True
except ImportError:
    print("SuperPoint not available. Skipping this method.")
    SUPERPOINT_AVAILABLE = False

try:
    from dloc.core.extractors.aliked import ALIKED
    ALIKED_AVAILABLE = True
except ImportError:
    print("ALIKED not available. Skipping this method.")
    ALIKED_AVAILABLE = False

try:
    from dloc.core.matchers.se2_loftr import SE2LoFTR
    SE2_LOFTR_AVAILABLE = True
except ImportError:
    print("SE2-LoFTR not available. Skipping this method.")
    SE2_LOFTR_AVAILABLE = False

# Check SIFT availability
try:
    # Try different SIFT creation methods for different OpenCV versions
    if hasattr(cv2, 'SIFT_create'):
        cv2.SIFT_create()
        SIFT_AVAILABLE = True
        SIFT_METHOD = 'create'
    elif hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SIFT_create'):
        cv2.xfeatures2d.SIFT_create()
        SIFT_AVAILABLE = True
        SIFT_METHOD = 'xfeatures2d'
    else:
        print("SIFT not available in this OpenCV version.")
        SIFT_AVAILABLE = False
        SIFT_METHOD = None
except Exception as e:
    print(f"SIFT not available: {e}")
    SIFT_AVAILABLE = False
    SIFT_METHOD = None

# Basic utility functions from the original code
def preprocess_image(image, size):
    """Resize image to a fixed square size."""
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return resized

def rotate_image(image, angle):
    """Rotate an image by a given angle in degrees."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # Compute new dimensions
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # Adjust transformation matrix
    rotation_matrix[0, 2] += bound_w / 2 - center[0]
    rotation_matrix[1, 2] += bound_h / 2 - center[1]

    # Perform rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))

    return rotated, rotation_matrix

def calculate_reprojection_error(kp1, kp2, matches, rot_matrix):
    """Calculate reprojection error for matches using the known rotation matrix."""
    errors = []

    for match in matches:
        # Get coordinates
        pt1 = np.array([kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1], 1.0])
        pt2 = np.array([kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1]])

        # Project point1 using rotation matrix
        projected_pt = np.dot(rot_matrix, pt1)[:2]

        # Calculate error
        error = np.linalg.norm(projected_pt - pt2)
        errors.append(error)

    return errors

def ensure_scode_input_size(image, target_size):
    """Ensure image meets size requirements for SCoDe (multiple of 32 and square)."""
    # Convert to RGB if needed
    if len(image.shape) == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize to target size
    resized = cv2.resize(rgb_image, (target_size, target_size))

    # Convert to tensor
    img_tensor = torch.from_numpy(resized).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    return img_tensor

def sample_images_from_pairs_file(dataset_path, pairs_file, num_images=36):
    """Sample unique images from the pairs file until we have enough."""
    unique_images = set()

    with open(pairs_file, 'r') as f:
        lines = f.readlines()

    # Shuffle the lines to get random samples
    random.shuffle(lines)

    for line in lines:
        if len(unique_images) >= num_images:
            break

        parts = line.strip().split()
        if len(parts) >= 3:  # Ensure there are enough parts for both image paths
            # Get paths for both images in the pair
            img1_path = os.path.join(dataset_path, parts[0])
            img2_path = os.path.join(dataset_path, parts[2])

            # Add first image if it exists and is not already included
            if os.path.exists(img1_path) and img1_path not in unique_images:
                unique_images.add(img1_path)

            # If we still need more images, add the second one
            if len(unique_images) < num_images and os.path.exists(img2_path) and img2_path not in unique_images:
                unique_images.add(img2_path)

    image_list = list(unique_images)
    print(f"Sampled {len(image_list)} unique images from the pairs file.")

    # Ensure we don't return more than requested
    return image_list[:num_images]

def sample_image_pairs_from_file(dataset_path, pairs_file, num_pairs=36, min_scale_ratio=2.0):
    """Sample image pairs from the pairs file with scale ratio > min_scale_ratio."""
    image_pairs = []

    with open(pairs_file, 'r') as f:
        lines = f.readlines()

    # Shuffle the lines to get random samples
    random.shuffle(lines)

    for line in lines:
        if len(image_pairs) >= num_pairs:
            break

        parts = line.strip().split()
        if len(parts) >= 5:  # Ensure there are enough parts for both images and overlap regions
            # Get paths for both images in the pair
            img1_path = os.path.join(dataset_path, parts[0])
            img2_path = os.path.join(dataset_path, parts[2])

            # Get overlap regions
            overlap1 = parts[1].split(',')
            overlap2 = parts[3].split(',')

            if len(overlap1) == 4 and len(overlap2) == 4:
                # Convert to integers
                x1_1, y1_1, x2_1, y2_1 = map(int, overlap1)
                x1_2, y1_2, x2_2, y2_2 = map(int, overlap2)

                # Calculate diagonal lengths
                diag1 = np.sqrt((x2_1 - x1_1)**2 + (y2_1 - y1_1)**2)
                diag2 = np.sqrt((x2_2 - x1_2)**2 + (y2_2 - y1_2)**2)

                # Calculate scale ratio
                scale_ratio = max(diag2 / diag1, diag1 / diag2) if min(diag1, diag2) > 0 else 1.0

                # Add pair if both images exist and scale ratio meets the minimum
                if os.path.exists(img1_path) and os.path.exists(img2_path) and scale_ratio > min_scale_ratio:
                    image_pairs.append((img1_path, img2_path, scale_ratio))

    print(f"Sampled {len(image_pairs)} image pairs with scale ratio > {min_scale_ratio} from the pairs file.")

    # Ensure we don't return more than requested
    return [(img1, img2) for img1, img2, _ in image_pairs[:num_pairs]]

# Feature extraction methods
class FeatureExtractor:
    def __init__(self, name, model=None):
        self.name = name
        self.model = model

    def extract(self, image):
        """Extract features from an image"""
        if self.name == 'orb':
            orb = cv2.ORB_create()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            return keypoints, descriptors

        elif self.name == 'sift' and SIFT_AVAILABLE:
            # Handle different OpenCV SIFT implementations
            if SIFT_METHOD == 'create':
                sift = cv2.SIFT_create()
            elif SIFT_METHOD == 'xfeatures2d':
                sift = cv2.xfeatures2d.SIFT_create()
            else:
                print("SIFT method not available")
                return [], None

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            return keypoints, descriptors

        elif self.name == 'disk' and DISK_AVAILABLE:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Get original dimensions
            orig_h, orig_w = rgb_image.shape[:2]

            # Calculate new dimensions (multiples of 16)
            new_h = ((orig_h + 15) // 16) * 16
            new_w = ((orig_w + 15) // 16) * 16

            # Resize image to dimensions compatible with DISK's U-Net
            resized_img = cv2.resize(rgb_image, (new_w, new_h))

            # Convert to tensor format expected by DISK
            img_tensor = torch.from_numpy(resized_img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # Extract features using DISK
            with torch.no_grad():
                features = self.model.features(img_tensor,
                                          kind='nms',
                                          window_size=5,
                                          cutoff=None,
                                          n=1024)
                features = features[0]  # Get first (only) element from the batch

            # Scale keypoints back to original image dimensions
            scale_x = orig_w / new_w
            scale_y = orig_h / new_h

            # Convert to OpenCV keypoints format with proper scaling
            cv_keypoints = []
            descriptors_list = []

            for i in range(features.kp.shape[0]):
                x, y = features.kp[i].cpu().numpy()
                # Scale back to original image coordinates
                x_orig = x * scale_x
                y_orig = y * scale_y
                response = features.kp_logp[i].cpu().item() if features.kp_logp is not None else 1.0

                # Handle different OpenCV versions (older versions use different parameter names)
                try:
                    # For newer OpenCV versions
                    cv_keypoints.append(cv2.KeyPoint(x=float(x_orig), y=float(y_orig), size=8.0, response=float(response)))
                except TypeError:
                    # For older OpenCV versions
                    cv_keypoints.append(cv2.KeyPoint(float(x_orig), float(y_orig), 8.0, -1, float(response), 0, -1))

                descriptors_list.append(features.desc[i].cpu().numpy())

            descriptors = np.array(descriptors_list)
            return cv_keypoints, descriptors

        elif self.name == 'aliked' and ALIKED_AVAILABLE:
            # ALIKED expects RGB image tensor format
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Convert to tensor format expected by ALIKED (NCHW)
            img_tensor = torch.from_numpy(rgb_image).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # Extract features using ALIKED
            with torch.no_grad():
                result = self.model({'image': img_tensor})

            # Get keypoints and descriptors from the result
            keypoints = result['keypoints'][0].cpu().numpy()
            scores = result['scores'][0].cpu().numpy()
            descriptors = result['descriptors'][0].cpu().numpy()

            # Convert to OpenCV keypoint format
            cv_keypoints = []
            for i in range(len(keypoints)):
                x, y = keypoints[i]
                cv_keypoints.append(cv2.KeyPoint(float(x), float(y), 8.0, -1, float(scores[i]), 0, -1))

            return cv_keypoints, descriptors.T

        elif self.name == 'r2d2' and R2D2_AVAILABLE:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Process image for R2D2
            img_tensor = torch.from_numpy(rgb_image).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # Extract features using R2D2 - use the proper API
            with torch.no_grad():
                # Call the model with a dictionary containing the image, following the standard pattern
                result = self.model({'image': img_tensor})

            # Get keypoints and descriptors directly from the result
            keypoints = result['keypoints'][0].cpu().numpy()
            scores = result['scores'][0].cpu().numpy()
            descriptors = result['descriptors'][0].cpu().numpy()

            # Convert to OpenCV keypoint format
            cv_keypoints = []
            for i in range(len(keypoints)):
                x, y = keypoints[i]
                cv_keypoints.append(cv2.KeyPoint(float(x), float(y), 8.0, -1, float(scores[i]), 0, -1))

            return cv_keypoints, descriptors.T

        elif self.name == 'd2net' and D2NET_AVAILABLE:
            # D2-Net expects RGB image for proper normalization
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Process image for D2-Net - properly format tensor
            img_tensor = torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # Extract features using D2-Net using the standard API
            with torch.no_grad():
                result = self.model({'image': img_tensor})

            # Get keypoints and descriptors from the result
            keypoints = result['keypoints'][0].cpu().numpy()
            scores = result['scores'][0].cpu().numpy()
            descriptors = result['descriptors'][0].cpu().numpy()

            # Convert keypoints to OpenCV format
            cv_keypoints = []
            for i in range(len(keypoints)):
                x, y = keypoints[i]  # D2Net returns keypoints in (x, y) format
                cv_keypoints.append(cv2.KeyPoint(float(x), float(y), 8.0, -1, float(scores[i]), 0, -1))

            return cv_keypoints, descriptors.T

        elif self.name == 'superpoint':
            # Prepare image for SuperPoint
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Create tensor
            img_tensor = torch.from_numpy(gray_image).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # Run SuperPoint
            with torch.no_grad():
                pred = self.model({'image': img_tensor})

            # Get keypoints and descriptors
            keypoints = pred['keypoints'][0].cpu().numpy()
            scores = pred['scores'][0].cpu().numpy()
            descriptors = pred['descriptors'][0].cpu().numpy()

            # Create OpenCV keypoints
            cv_keypoints = []
            for i in range(len(keypoints)):
                x, y = keypoints[i]
                cv_keypoints.append(cv2.KeyPoint(float(x), float(y), 8.0, -1, float(scores[i]), 0, -1))

            return cv_keypoints, descriptors.T

        elif self.name == 'se2_loftr' and SE2_LOFTR_AVAILABLE:
            # SE2-LoFTR expects grayscale images
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Convert to tensor format
            img_tensor = torch.from_numpy(gray_image).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # For SE2-LoFTR, we need both images to perform matching
            # This is a limitation for single image extraction
            # Return empty results as SE2-LoFTR is an end-to-end matcher
            print("Warning: SE2-LoFTR is an end-to-end matcher and cannot extract features from single images.")
            return [], None

        return [], None

# Feature Matcher classes
class FeatureMatcher:
    def __init__(self, name):
        self.name = name

    def match(self, kp1, desc1, kp2, desc2):
        """Match features between two images"""
        if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
            return []

        if self.name == 'flann':
            # Convert descriptors to appropriate type if needed
            if desc1.dtype != np.float32:
                desc1 = desc1.astype(np.float32)
                desc2 = desc2.astype(np.float32)

            flann_params = dict(algorithm=1, trees=5)
            matcher = cv2.FlannBasedMatcher(flann_params, {})

            # Perform matching
            matches = matcher.knnMatch(desc1, desc2, k=2)

            # Apply ratio test
            good_matches = []
            for matches_for_descriptor in matches:
                if len(matches_for_descriptor) == 2:  # Only perform ratio test if we have 2 matches
                    m, n = matches_for_descriptor
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            return good_matches

        elif self.name == 'bf':
            # Convert descriptors to appropriate type if needed
            if desc1.dtype != np.float32:
                desc1 = desc1.astype(np.float32)
                desc2 = desc2.astype(np.float32)

            # Brute force matching
            matcher = cv2.BFMatcher(cv2.NORM_L2)

            # Perform matching
            matches = matcher.knnMatch(desc1, desc2, k=2)

            # Apply ratio test
            good_matches = []
            for matches_for_descriptor in matches:
                if len(matches_for_descriptor) == 2:  # Only perform ratio test if we have 2 matches
                    m, n = matches_for_descriptor
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            return good_matches

# End-to-end Matcher class for SE2-LoFTR
class EndToEndMatcher:
    def __init__(self, name, model=None, overlap_estimator=None):
        self.name = name
        self.model = model
        self.overlap_estimator = overlap_estimator

    def match_images(self, img1, img2):
        """Direct end-to-end matching between two images"""
        if ('se2_loftr' in self.name) and self.model is not None:
            try:
                if self.overlap_estimator is not None:
                    overlap_img1, overlap_img2, scale1, scale2 = self.overlap_estimator.predict_overlap(img1, img2, 640)
                    img1_to_use = overlap_img1
                    img2_to_use = overlap_img2
                else:
                    img1_to_use = img1
                    img2_to_use = img2
                
                target_size = 640
                img1_resized = cv2.resize(img1_to_use, (target_size, target_size))
                img2_resized = cv2.resize(img2_to_use, (target_size, target_size))
                
                # Convert images to tensor format
                if len(img1_resized.shape) == 3:
                    gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
                else:
                    gray1 = img1_resized

                if len(img2_resized.shape) == 3:
                    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
                else:
                    gray2 = img2_resized

                # Convert to tensors
                img1_tensor = torch.from_numpy(gray1).float() / 255.0
                img1_tensor = img1_tensor.unsqueeze(0).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                img2_tensor = torch.from_numpy(gray2).float() / 255.0
                img2_tensor = img2_tensor.unsqueeze(0).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                # Perform matching
                with torch.no_grad():
                    result = self.model({'image0': img1_tensor, 'image1': img2_tensor})

                # Extract results
                mkpts0 = result['keypoints0'][0].cpu().numpy()
                mkpts1 = result['keypoints1'][0].cpu().numpy()
                mconf = result['matching_scores0'][0].cpu().numpy()

                # Scale keypoints back if using SCoDe
                if self.overlap_estimator is not None:
                    scale_x1, scale_y1 = scale1
                    scale_x2, scale_y2 = scale2
                    # Scale keypoints back to original image coordinates
                    mkpts0 *= [scale_x1, scale_y1]
                    mkpts1 *= [scale_x2, scale_y2]
                else:
                    # Scale from resized to original
                    scale_x = img1.shape[1] / target_size
                    scale_y = img1.shape[0] / target_size
                    mkpts0 *= [scale_x, scale_y]
                    mkpts1 *= [scale_x, scale_y]

                # Create dummy matches structure for compatibility
                matches = []
                for i in range(len(mkpts0)):
                    match = type('DMatch', (), {
                        'queryIdx': i,
                        'trainIdx': i,
                        'distance': 1.0 - mconf[i] if len(mconf) > i else 0.5
                    })()
                    matches.append(match)

                # Create dummy keypoints for compatibility
                kp1 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 8.0) for pt in mkpts0]
                kp2 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 8.0) for pt in mkpts1]

                return kp1, kp2, matches
                
            except Exception as e:
                print(f"Error in SE2-LoFTR matching: {e}")
                return [], [], []

        return [], [], []

# Overlap Estimator class
class OverlapEstimator:
    def __init__(self, name, model=None):
        self.name = name
        self.model = model

    def predict_overlap(self, img1, img2=None, image_size=640, return_bboxes=False):
        """Predict overlap region between two images and resize to a fixed size.
        If return_bboxes=True, also return (bbox0, bbox1) in original coordinates.
        """
        if self.name == 'scode' and self.model is not None:
            img1_tensor = ensure_scode_input_size(img1, image_size)
            img2_tensor = ensure_scode_input_size(img2, image_size) if img2 is not None else img1_tensor.clone()

            try:
                bbox0, bbox1 = self.model({'image0': img1_tensor, 'image1': img2_tensor})

                x_min = max(0, int(bbox0[0][0].item()))
                y_min = max(0, int(bbox0[0][1].item()))
                x_max = min(img1.shape[1], int(bbox0[0][2].item()))
                y_max = min(img1.shape[0], int(bbox0[0][3].item()))

                overlap_img1 = img1[y_min:y_max, x_min:x_max]
                scale_x1 = (x_max - x_min) / image_size if (x_max - x_min) > 0 else 1.0
                scale_y1 = (y_max - y_min) / image_size if (y_max - y_min) > 0 else 1.0
                overlap_img1_resized = cv2.resize(overlap_img1, (image_size, image_size)) if overlap_img1.size > 0 else cv2.resize(img1, (image_size, image_size))

                if img2 is not None:
                    x_min2 = max(0, int(bbox1[0][0].item()))
                    y_min2 = max(0, int(bbox1[0][1].item()))
                    x_max2 = min(img2.shape[1], int(bbox1[0][2].item()))
                    y_max2 = min(img2.shape[0], int(bbox1[0][3].item()))

                    overlap_img2 = img2[y_min2:y_max2, x_min2:x_max2]
                    scale_x2 = (x_max2 - x_min2) / image_size if (x_max2 - x_min2) > 0 else 1.0
                    scale_y2 = (y_max2 - y_min2) / image_size if (y_max2 - y_min2) > 0 else 1.0
                    overlap_img2_resized = cv2.resize(overlap_img2, (image_size, image_size)) if overlap_img2.size > 0 else cv2.resize(img2, (image_size, image_size))
                else:
                    overlap_img2_resized = overlap_img1_resized
                    scale_x2, scale_y2 = scale_x1, scale_y1
                    x_min2, y_min2, x_max2, y_max2 = x_min, y_min, x_max, y_max

                if return_bboxes:
                    return (overlap_img1_resized, overlap_img2_resized,
                            (scale_x1, scale_y1), (scale_x2, scale_y2),
                            (x_min, y_min, x_max, y_max), (x_min2, y_min2, x_max2, y_max2))
                return overlap_img1_resized, overlap_img2_resized, (scale_x1, scale_y1), (scale_x2, scale_y2)
            except Exception as e:
                print(f"Warning: Overlap prediction failed: {e}")

        # Fallback: whole image; fabricate bbox if requested
        if return_bboxes:
            h1, w1 = img1.shape[:2]
            if img2 is not None:
                h2, w2 = img2.shape[:2]
                return img1, img2, (1.0, 1.0), (1.0, 1.0), (0, 0, w1, h1), (0, 0, w2, h2)
            return img1, img1, (1.0, 1.0), (1.0, 1.0), (0, 0, w1, h1), (0, 0, w1, h1)
        return img1, img2 if img2 is not None else img1, (1.0, 1.0), (1.0, 1.0)

def filter_keypoints_by_region(keypoints, descriptors, region):
    """Filter keypoints to only include those within the specified region"""
    x_min, y_min, x_max, y_max = region
    filtered_keypoints = []
    filtered_desc_indices = []

    # Check each keypoint to see if it falls within the region
    for i, kp in enumerate(keypoints):
        x, y = kp.pt
        if x_min <= x <= x_max and y_min <= y <= y_max:
            filtered_keypoints.append(kp)
            filtered_desc_indices.append(i)

    # Filter descriptors if they exist
    filtered_descriptors = None
    if descriptors is not None and len(filtered_desc_indices) > 0:
        filtered_descriptors = descriptors[filtered_desc_indices]

    return filtered_keypoints, filtered_descriptors

def evaluate_rotation_invariance_pipeline(image_pairs, pipelines, angles=range(0, 50, 1), threshold=3.0, image_size=640):
    """
    Evaluate rotation invariance of different pipelines using the number of valid matches as a metric.
    """
    # Initialize results dictionary
    results = {pipeline[3]: {angle: [] for angle in angles} for pipeline in pipelines}

    # Process each image pair
    for img1_path, img2_path in tqdm(image_pairs, desc="Processing image pairs"):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"Failed to load images: {img1_path}, {img2_path}")
            continue

        # Standardize image size
        img1 = preprocess_image(img1, size=image_size)
        img2 = preprocess_image(img2, size=image_size)

        # Process each pipeline
        for extractor, overlap_estimator, matcher, pipeline_name in pipelines:
            kp1, desc1 = extractor.extract(img1)

            for angle in angles:
                rotated_img2, rot_matrix = rotate_image(img2, angle)

                if overlap_estimator:
                    overlap_img1, overlap_img2, scale1, scale2 = overlap_estimator.predict_overlap(img1, rotated_img2, image_size)
                    kp2_rot, desc2_rot = extractor.extract(overlap_img2)

                    # Map keypoints back to original image coordinates (vectorized)
                    if kp2_rot:
                        scale_x, scale_y = scale2
                        for kp in kp2_rot:
                            kp.pt = (kp.pt[0] * scale_x, kp.pt[1] * scale_y)
                else:
                    kp2_rot, desc2_rot = extractor.extract(rotated_img2)

                matches = matcher.match(kp1, desc1, kp2_rot, desc2_rot)

                # Apply RANSAC to filter matches
                if len(matches) > 4:  # Need at least 4 points for homography
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2_rot[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    # Use RANSAC with adjusted parameters for better robustness
                    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
                    inlier_count = np.sum(mask) if mask is not None else 0
                    match_ratio = inlier_count / len(matches)
                else:
                    match_ratio = 0.0

                results[pipeline_name][angle].append(match_ratio)

    # Calculate average match ratio for each angle
    for pipeline_name in results:
        for angle in angles:
            if results[pipeline_name][angle]:
                results[pipeline_name][angle] = np.mean(results[pipeline_name][angle])
            else:
                results[pipeline_name][angle] = 0.0

    return results


def plot_pipeline_results(results, dataset_name, output_dir='./outputs/scode_rot_eval', show_raw_points=False):
    """Plot rotation invariance evaluation results for different pipelines."""
    os.makedirs(output_dir, exist_ok=True)

    pipeline_names = list(results.keys())
    angles = sorted(list(results[pipeline_names[0]].keys()))

    # Adjust figure size to make it wider and flatter
    plt.figure(figsize=(16, 7))

    # Define base colors for different methods
    base_colors = {
        'DISK': '#1f77b4',
        'R2D2': '#ff7f0e',
        'D2-Net': '#2ca02c',
        'SuperPoint': '#d62728',
        'SIFT': '#9467bd',
        'ALIKED': '#8c564b',
        'SE2-LoFTR': '#e377c2'
    }

    # Create denser angle points for smoother plotting
    fine_angles = np.linspace(min(angles), max(angles), num=500)

    for pipeline_name in pipeline_names:
        # Determine base color and adjust for SCoDe
        if 'SE2-LoFTR' in pipeline_name:
            method_name = 'SE2-LoFTR'
            is_scode = '+SCoDe' in pipeline_name
        else:
            method_name = pipeline_name.split('-')[0]
            is_scode = '+SCoDe' in pipeline_name
        
        base_color = base_colors.get(method_name, '#7f7f7f')

        # Make non-SCoDe curves thinner and more transparent
        if is_scode:
            color = base_color
            linewidth = 2.8
            alpha = 1.0
            marker_size = 6
        else:
            color = base_color
            linewidth = 1.8
            alpha = 0.5
            marker_size = 4

        # Extract raw data
        match_ratios = [results[pipeline_name][angle] for angle in angles]

        # Plot raw data points if requested
        if show_raw_points:
            plt.scatter(angles, match_ratios, color=color, alpha=alpha*0.8, 
                       s=marker_size*8, zorder=5, edgecolors='white', linewidth=0.5)

        # Use polynomial fitting to get smooth curves
        poly_degree = 8
        if poly_degree > 0:
            try:
                poly_coeffs = np.polyfit(angles, match_ratios, poly_degree)
                poly_func = np.poly1d(poly_coeffs)
                
                # Calculate polynomial values at dense angle points
                smooth_curve = poly_func(fine_angles)
                
                # Remove the near-zero flattening - just ensure no negative values
                smooth_curve = np.maximum(smooth_curve, 0)
                
                # Plot the smooth curve
                plt.plot(fine_angles, smooth_curve, color=color, linewidth=linewidth,
                         alpha=alpha, label=pipeline_name, zorder=3)
            except np.RankWarning:
                # If polynomial fitting fails, fall back to linear interpolation
                from scipy.interpolate import interp1d
                interp_func = interp1d(angles, match_ratios, kind='cubic', fill_value='extrapolate')
                smooth_curve = interp_func(fine_angles)
                smooth_curve = np.maximum(smooth_curve, 0)
                plt.plot(fine_angles, smooth_curve, color=color, linewidth=linewidth,
                         alpha=alpha, label=pipeline_name, zorder=3)
        else:
            # If we don't have enough points for polynomial fitting, just plot the raw data
            plt.plot(angles, match_ratios, color=color, linewidth=linewidth,
                     alpha=alpha, label=pipeline_name, marker='o')

    plt.xlabel('Rotation Angle (degrees)', fontsize=14)
    plt.ylabel('RANSAC-verified Match Ratio', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Set chart style
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save the plot
    suffix = '_with_raw_points' if show_raw_points else ''
    output_path = os.path.join(output_dir, f'scode_rotation_invariance_{dataset_name}{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Results saved to {output_path}")

def save_evaluation_results(results, dataset_name, output_dir='./outputs/scode_rot_eval'):
    """Save rotation invariance evaluation results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a filename based on dataset name
    output_path = os.path.join(output_dir, f'rotation_invariance_results_{dataset_name}.pkl')

    # Save results dictionary using pickle
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Evaluation results saved to {output_path}")
    return output_path

def load_evaluation_results(dataset_name, output_dir='./outputs/scode_rot_eval'):
    """Load rotation invariance evaluation results from disk."""
    # Create a filename based on dataset name
    input_path = os.path.join(output_dir, f'rotation_invariance_results_{dataset_name}.pkl')

    # Check if file exists
    if not os.path.exists(input_path):
        print(f"No saved results found at {input_path}")
        return None

    # Load results dictionary using pickle
    with open(input_path, 'rb') as f:
        results = pickle.load(f)

    print(f"Loaded evaluation results from {input_path}")
    return results

def visualize_matches(img1, img2, kp1, kp2, matches, angle, method_name, output_dir, pair_idx=0):
    """Visualize matches between two images."""
    # Create output visualization
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create side-by-side image
    vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    
    # Convert images to BGR if they're grayscale
    if len(img1.shape) == 2:
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_bgr = img1.copy()
    
    if len(img2.shape) == 2:
        img2_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_bgr = img2.copy()
    
    # Place images side by side
    vis_img[:h1, :w1] = img1_bgr
    vis_img[:h2, w1:w1+w2] = img2_bgr
    
    # Draw matches
    for i, match in enumerate(matches[:50]):  # Limit to first 50 matches for clarity
        # Get keypoint coordinates
        pt1 = (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1]))
        pt2 = (int(kp2[match.trainIdx].pt[0] + w1), int(kp2[match.trainIdx].pt[1]))
        
        # Generate color based on match index
        color = (
            int(255 * (i / len(matches))),
            int(255 * (1 - i / len(matches))),
            128
        )
        
        # Draw keypoints
        cv2.circle(vis_img, pt1, 3, color, -1)
        cv2.circle(vis_img, pt2, 3, color, -1)
        
        # Draw line connecting matches
        cv2.line(vis_img, pt1, pt2, color, 1)
    
    # Add text information
    text = f"{method_name} | Angle: {angle}° | Matches: {len(matches)}"
    cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save visualization
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    output_path = os.path.join(output_dir, 'visualizations', 
                              f'{method_name}_pair{pair_idx}_angle{angle}.png')
    cv2.imwrite(output_path, vis_img)
    
    return output_path

def visualize_overlap_debug(img1, img2_rot, bbox1, bbox2, angle, pair_idx, output_dir):
    """Visualize predicted overlap bounding boxes (debug mode)."""
    os.makedirs(output_dir, exist_ok=True)
    img1_vis = img1.copy()
    img2_vis = img2_rot.copy()
    if len(img1_vis.shape) == 2:
        img1_vis = cv2.cvtColor(img1_vis, cv2.COLOR_GRAY2BGR)
    if len(img2_vis.shape) == 2:
        img2_vis = cv2.cvtColor(img2_vis, cv2.COLOR_GRAY2BGR)

    (x1a, y1a, x1b, y1b) = bbox1
    (x2a, y2a, x2b, y2b) = bbox2
    cv2.rectangle(img1_vis, (x1a, y1a), (x1b, y1b), (0, 255, 0), 2)
    cv2.rectangle(img2_vis, (x2a, y2a), (x2b, y2b), (0, 255, 0), 2)
    cv2.putText(img1_vis, f'Angle {angle}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(img2_vis, f'Angle {angle}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Concatenate
    h = max(img1_vis.shape[0], img2_vis.shape[0])
    canvas = np.zeros((h, img1_vis.shape[1] + img2_vis.shape[1], 3), dtype=np.uint8)
    canvas[:img1_vis.shape[0], :img1_vis.shape[1]] = img1_vis
    canvas[:img2_vis.shape[0], img1_vis.shape[1]:] = img2_vis

    out_path = os.path.join(output_dir, f'pair{pair_idx}_angle{angle}.png')
    cv2.imwrite(out_path, canvas)
    return out_path

def assemble_debug_montage(image_paths, angles, output_dir, pair_idx):
    """Assemble all angle visualizations of one pair into a single montage image."""
    if not image_paths:
        return
    panels = []
    max_h = 0
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        panels.append(img)
        max_h = max(max_h, img.shape[0])
    if not panels:
        return
    # Pad to same height
    padded = []
    for img in panels:
        if img.shape[0] < max_h:
            pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
            img = np.vstack([img, pad])
        padded.append(img)
    montage = np.hstack(padded)
    cv2.putText(montage, f'Pair {pair_idx} Angles: {",".join(map(str, angles))}',
                (20, min(40, max_h-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    out_path = os.path.join(output_dir, f'pair{pair_idx}_montage.png')
    cv2.imwrite(out_path, montage)
    return out_path

def evaluate_rotation_invariance_pipeline_with_end2end(image_pairs, pipelines, end2end_matchers, angles=range(0, 50, 1), threshold=3.0, image_size=640, visualize=False, vis_angles=None, output_dir='./outputs/scode_rot_eval'):
    """
    Enhanced evaluation function that handles both traditional and end-to-end matching pipelines.
    
    Args:
        visualize: Whether to save match visualizations
        vis_angles: List of specific angles to visualize (if None, visualize all angles)
        output_dir: Directory to save visualizations
    """
    # Initialize results dictionary for both traditional and end-to-end methods
    results = {pipeline[3]: {angle: [] for angle in angles} for pipeline in pipelines}
    for matcher_name in end2end_matchers:
        results[matcher_name] = {angle: [] for angle in angles}

    # Determine which angles to visualize
    if visualize and vis_angles is None:
        vis_angles = [0, 30, 60, 90, 120, 150]  # Default visualization angles
    elif visualize:
        vis_angles = vis_angles
    else:
        vis_angles = []

    # Process each image pair
    for pair_idx, (img1_path, img2_path) in enumerate(tqdm(image_pairs, desc="Processing image pairs")):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"Failed to load images: {img1_path}, {img2_path}")
            continue

        # Standardize image size
        img1 = preprocess_image(img1, size=image_size)
        img2 = preprocess_image(img2, size=image_size)

        # Process traditional pipelines
        for extractor, overlap_estimator, matcher, pipeline_name in pipelines:
            kp1, desc1 = extractor.extract(img1)

            for angle in angles:
                rotated_img2, rot_matrix = rotate_image(img2, angle)

                if overlap_estimator:
                    overlap_img1, overlap_img2, scale1, scale2 = overlap_estimator.predict_overlap(img1, rotated_img2, image_size)
                    kp2_rot, desc2_rot = extractor.extract(overlap_img2)

                    # Map keypoints back to original image coordinates (vectorized)
                    if kp2_rot:
                        scale_x, scale_y = scale2
                        for kp in kp2_rot:
                            kp.pt = (kp.pt[0] * scale_x, kp.pt[1] * scale_y)
                else:
                    kp2_rot, desc2_rot = extractor.extract(rotated_img2)

                matches = matcher.match(kp1, desc1, kp2_rot, desc2_rot)

                # Visualize matches if requested
                if visualize and angle in vis_angles and pair_idx < 3:  # Limit to first 3 pairs
                    try:
                        visualize_matches(img1, rotated_img2, kp1, kp2_rot, matches, 
                                        angle, pipeline_name, output_dir, pair_idx)
                    except Exception as e:
                        print(f"Warning: Failed to visualize matches for {pipeline_name}: {e}")

                # Apply RANSAC to filter matches
                if len(matches) > 4:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2_rot[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
                    inlier_count = np.sum(mask) if mask is not None else 0
                    match_ratio = inlier_count / len(matches)
                else:
                    match_ratio = 0.0

                results[pipeline_name][angle].append(match_ratio)

        # Process end-to-end matchers
        for matcher_name, matcher in end2end_matchers.items():
            for angle in angles:
                rotated_img2, rot_matrix = rotate_image(img2, angle)

                # Direct end-to-end matching
                kp1, kp2_rot, matches = matcher.match_images(img1, rotated_img2)

                # Visualize matches if requested
                if visualize and angle in vis_angles and pair_idx < 3:  # Limit to first 3 pairs
                    try:
                        visualize_matches(img1, rotated_img2, kp1, kp2_rot, matches, 
                                        angle, matcher_name, output_dir, pair_idx)
                    except Exception as e:
                        print(f"Warning: Failed to visualize matches for {matcher_name}: {e}")

                # Apply RANSAC to filter matches
                if len(matches) > 4:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2_rot[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
                    inlier_count = np.sum(mask) if mask is not None else 0
                    match_ratio = inlier_count / len(matches)
                else:
                    match_ratio = 0.0

                results[matcher_name][angle].append(match_ratio)

    # Calculate average match ratio for each angle
    for method_name in results:
        for angle in angles:
            if results[method_name][angle]:
                results[method_name][angle] = np.mean(results[method_name][angle])
            else:
                results[method_name][angle] = 0.0

    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate SCoDe rotation invariance across different matching pipelines')
    parser.add_argument('--dataset', type=str, default='/data/nfs/lhj/OverlapEstimation/SCoDe/dataset/megadepth/MegaDepth',
                        help='Path to dataset')
    parser.add_argument('--dataset_name', type=str, default='MegaDepth',
                        help='Dataset name for plot title')
    parser.add_argument('--num_images', type=int, default=36,
                        help='Number of individual images to select')
    parser.add_argument('--pairs_file', type=str, default='dataset/megadepth/assets/MegaDepth_from_scale_valid_V2.txt',
                        help='Path to file containing image pairs to sample from')
    parser.add_argument('--angle_step', type=int, default=10,
                        help='Step size for rotation angles')
    parser.add_argument('--output_dir', type=str, default='./outputs/scode_rot_eval',
                        help='Output directory for results')
    parser.add_argument('--image_size', type=int, default=640,
                        help='Size for square image standardization')
    parser.add_argument('--extractors', type=str, nargs='+',
                        default=['se2_loftr', 'sift', 'aliked', 'disk', 'r2d2', 'd2net', 'superpoint'],
                        help='Feature extractors to use')
    parser.add_argument('--end2end_matchers', type=str, nargs='+',
                        default=['se2_loftr'],
                        help='End-to-end matchers to use')
    parser.add_argument('--matchers', type=str, nargs='+', default=['bf'],
                        help='Feature matchers to use')
    parser.add_argument('--force_recompute', action='store_true',
                        help='Force recomputation of results even if saved data exists')
    parser.add_argument('--plot_only', action='store_true',
                        help='Only plot results from saved data without checking models')
    parser.add_argument('--visualize', action='store_true',
                        help='Save match visualizations for selected angles')
    parser.add_argument('--vis_angles', type=int, nargs='+', default=[0, 30, 60, 90, 120, 150],
                        help='Specific angles to visualize (default: 0, 30, 60, 90, 120, 150)')
    parser.add_argument('--show_raw_points', action='store_true',
                        help='Show raw data points on the plot before curve fitting')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode: only run overlap detection & visualization')
    args = parser.parse_args()

    # Debug mode: only overlap detection visualization
    if args.debug:
        print("Running in DEBUG mode: only overlap detection and visualization.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights_path = Path('/data/nfs/lhj/OverlapEstimation/SCoDe/weights/')
        # Load SCoDe model only
        try:
            scode_config = {
                'name': 'scode',
                'model': 'ccoe',
                'num_layers': 50,
                'stride': 32,
                'last_layer': 1024,
                'layer': 'layer3',
                'weights': 'scode_1024_pt4.pth',
            }
            scode_model = SCoDe(scode_config, weights_path).to(device).eval()
            overlap_estimator = OverlapEstimator('scode', scode_model)
            print("SCoDe model loaded for debug.")
        except Exception as e:
            print(f"Failed to load SCoDe model: {e}")
            return

        # Load image pairs
        if args.pairs_file and os.path.exists(args.pairs_file):
            image_pairs = sample_image_pairs_from_file(args.dataset, args.pairs_file, args.num_images)
        else:
            print("Error: Pairs file not found.")
            return

        print(f"Debug: {len(image_pairs)} image pairs loaded.")
        debug_output_dir = os.path.join('./outputs', 'debug')
        os.makedirs(debug_output_dir, exist_ok=True)

        # Key angles (0,30,60,90,... <180)
        angles = list(range(0, 180, 30))
        for pair_idx, (img1_path, img2_path) in enumerate(tqdm(image_pairs, desc="Debug overlap")):
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            if img1 is None or img2 is None:
                print(f"Failed to load: {img1_path}, {img2_path}")
                continue
            img1 = preprocess_image(img1, size=args.image_size)
            img2 = preprocess_image(img2, size=args.image_size)

            per_angle_paths = []
            for angle in angles:
                rotated_img2, _ = rotate_image(img2, angle)
                # Get overlap with bboxes
                (over1_resized, over2_resized, _, _, bbox1, bbox2) = overlap_estimator.predict_overlap(
                    img1, rotated_img2, args.image_size, return_bboxes=True)
                path = visualize_overlap_debug(img1, rotated_img2, bbox1, bbox2, angle, pair_idx, debug_output_dir)
                per_angle_paths.append(path)
            assemble_debug_montage(per_angle_paths, angles, debug_output_dir, pair_idx)
        print(f"Debug outputs saved to {debug_output_dir}")
        return

    # Check if saved results already exist
    results = None
    if not args.force_recompute:
        results = load_evaluation_results(args.dataset_name, args.output_dir)

    # If plot_only is specified and results exist, skip directly to plotting
    if args.plot_only and results is not None:
        print("Plot only mode: Using saved results for plotting.")
        plot_pipeline_results(results, args.dataset_name, args.output_dir, args.show_raw_points)
        return

    # If no saved results or force_recompute is True, run the evaluation
    if results is None:
        image_size = args.image_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights_path = Path('/data/nfs/lhj/OverlapEstimation/SCoDe/weights/')

        # Define standard model configurations
        model_configs = {
            'disk': {
                'output': 'feats-disk-desc',
                'model': {
                    'name': 'disk',
                    'window': 8,
                    'desc_dim': 128,
                    'weights': 'disk/disk_depth.pth',
                },
                'preprocessing': {
                    'grayscale': False,
                    'resize_max': 1280,
                },
            },
            'sift': {
                'output': 'feats-sift-desc',
                'model': {
                    'name': 'sift',
                    'max_keypoints': 2048,
                },
                'preprocessing': {
                    'grayscale': True,
                    'resize_max': 1280,
                },
            },
            'aliked': {
                'output': 'feats-aliked-desc',
                'model': {
                    'name': 'aliked',
                    'model_name': 'aliked-n16',
                    'max_keypoints': 2048,
                    'detection_threshold': 0.2,
                    'weights': 'aliked/aliked-n16rot.pth',
                },
                'preprocessing': {
                    'grayscale': False,
                    'resize_max': 1280,
                },
            },
            'r2d2': {
                'output': 'feats-r2d2-desc',
                'model': {
                    'name': 'r2d2',
                    'model': 'r2d2_WASF_N16.pt',
                    'reliability_thr': 0.7,
                    'repeatability_thr': 0.7,
                    'topk': 5000,
                },
                'preprocessing': {
                    'grayscale': False,
                    'resize_max': 1024,
                },
            },
            'd2net': {
                'output': 'feats-d2net-ss',
                'model': {
                    'name': 'd2net',
                    'model_name': 'd2_tf.pth',
                    'use_relu': True,
                    'multiscale': False,
                },
                'preprocessing': {
                    'grayscale': False,
                    'resize_max': 1600,
                },
            },
            'superpoint': {
                'output': 'feats-superpoint-n4096',
                'model': {
                    'name': 'superpoint',
                    'nms_radius': 4,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': 1024,
                    'remove_borders': 4,
                    'model_path': 'superpoint_v1.pth',
                },
                'preprocessing': {
                    'grayscale': True,
                    'resize_max': 1600,
                },
            },
            'scode': {
                'output': 'preds-scode',
                'model': {
                    'name': 'scode',
                    'model': 'ccoe',
                    'num_layers': 50,
                    'stride': 32,
                    'last_layer': 1024,
                    'layer': 'layer3',
                    'weights': 'scode.pth',
                },
                'preprocessing': {
                    'grayscale': False,
                    'resize_max': 640,
                },
            },
            'se2_loftr': {
                'output': 'feats-se2-loftr',
                'model': {
                    'name': 'se2_loftr',
                    'weights': 'se2_loftr/4rot-big.ckpt',
                    'match_threshold': 0.2,
                    'max_keypoints': 2048,
                },
                'preprocessing': {
                    'grayscale': True,
                    'resize_max': 1024,
                },
            },
        }

        # Initialize models using a consistent approach
        extractors = {}
        matchers = {'bf': FeatureMatcher('bf')}

        # SIFT model initialization (check availability first)
        if 'sift' in args.extractors:
            if SIFT_AVAILABLE:
                extractors['sift'] = FeatureExtractor('sift', None)
                print("SIFT extractor initialized successfully")
            else:
                print("SIFT requested but not available. Skipping SIFT.")

        # ALIKED model initialization
        if 'aliked' in args.extractors:
            if ALIKED_AVAILABLE:
                try:
                    aliked_config = model_configs['aliked']['model']
                    aliked_model = ALIKED(aliked_config, weights_path)
                    aliked_model = aliked_model.to(device).eval()
                    extractors['aliked'] = FeatureExtractor('aliked', aliked_model)
                    print("ALIKED extractor initialized successfully")
                except Exception as e:
                    print(f"Error loading ALIKED model: {e}")
            else:
                print("ALIKED requested but not available. Skipping ALIKED.")

        # DISK model initialization
        if 'disk' in args.extractors and DISK_AVAILABLE:
            try:
                disk_config = model_configs['disk']['model']
                disk_model = DISK(window=disk_config['window'], desc_dim=disk_config['desc_dim'])
                checkpoint_path = weights_path / disk_config['weights']

                if checkpoint_path.exists():
                    state_dict = torch.load(str(checkpoint_path), map_location='cpu')
                    weights = state_dict.get('extractor', state_dict)
                    disk_model.load_state_dict(weights)
                    disk_model = disk_model.to(device).eval()
                    extractors['disk'] = FeatureExtractor('disk', disk_model)
                    print("DISK extractor initialized successfully")
                else:
                    print(f"DISK checkpoint not found at {checkpoint_path}")
            except Exception as e:
                print(f"Error loading DISK model: {e}")

        # R2D2 model initialization
        if 'r2d2' in args.extractors and R2D2_AVAILABLE:
            try:
                r2d2_config = model_configs['r2d2']['model']
                r2d2_model = R2D2Desc(r2d2_config, weights_path)
                r2d2_model = r2d2_model.to(device).eval()
                extractors['r2d2'] = FeatureExtractor('r2d2', r2d2_model)
                print("R2D2 extractor initialized successfully")
            except Exception as e:
                print(f"Error loading R2D2 model: {e}")

        # D2-Net model initialization
        if 'd2net' in args.extractors and D2NET_AVAILABLE:
            try:
                d2net_config = model_configs['d2net']['model']
                d2net_model = D2Net(d2net_config, weights_path)
                d2net_model = d2net_model.to(device).eval()
                extractors['d2net'] = FeatureExtractor('d2net', d2net_model)
                print("D2-Net extractor initialized successfully")
            except Exception as e:
                print(f"Error loading D2-Net model: {e}")

        # SuperPoint model initialization
        if 'superpoint' in args.extractors and SUPERPOINT_AVAILABLE:
            try:
                superpoint_config = model_configs['superpoint']['model']
                superpoint_model = SuperPoint(superpoint_config, weights_path)
                superpoint_model = superpoint_model.to(device).eval()
                extractors['superpoint'] = FeatureExtractor('superpoint', superpoint_model)
                print("SuperPoint extractor initialized successfully")
            except Exception as e:
                print(f"Error loading SuperPoint model: {e}")

        # Initialize SCoDe model (SCoDe)
        scode_model = None
        try:
            scode_config = model_configs['scode']['model']
            scode_model = SCoDe(scode_config, weights_path)
            scode_model = scode_model.to(device).eval()
            overlap_estimator = OverlapEstimator('scode', scode_model)
            print("SCoDe (SCoDe) model initialized successfully")
        except Exception as e:
            print(f"Error loading SCoDe model: {e}")
            print("SCoDe will not be available for evaluation.")
            overlap_estimator = None

        # Define pipelines as (extractor, overlap_estimator, matcher, name)
        pipelines = []

        # Add baseline methods without SCoDe
        if 'sift' in extractors:
            pipelines.append((extractors['sift'], None, matchers['bf'], 'SIFT-BF'))
        if 'aliked' in extractors:
            pipelines.append((extractors['aliked'], None, matchers['bf'], 'ALIKED-BF'))
        if 'disk' in extractors:
            pipelines.append((extractors['disk'], None, matchers['bf'], 'DISK-BF'))
        if 'r2d2' in extractors:
            pipelines.append((extractors['r2d2'], None, matchers['bf'], 'R2D2-BF'))
        if 'd2net' in extractors:
            pipelines.append((extractors['d2net'], None, matchers['bf'], 'D2-Net-BF'))
        if 'superpoint' in extractors:
            pipelines.append((extractors['superpoint'], None, matchers['bf'], 'SuperPoint-BF'))

        # Add methods with SCoDe if available
        if overlap_estimator is not None:
            if 'sift' in extractors:
                pipelines.append((extractors['sift'], overlap_estimator, matchers['bf'], 'SIFT-BF+SCoDe'))
            if 'aliked' in extractors:
                pipelines.append((extractors['aliked'], overlap_estimator, matchers['bf'], 'ALIKED-BF+SCoDe'))
            if 'disk' in extractors:
                pipelines.append((extractors['disk'], overlap_estimator, matchers['bf'], 'DISK-BF+SCoDe'))
            if 'r2d2' in extractors:
                pipelines.append((extractors['r2d2'], overlap_estimator, matchers['bf'], 'R2D2-BF+SCoDe'))
            if 'd2net' in extractors:
                pipelines.append((extractors['d2net'], overlap_estimator, matchers['bf'], 'D2-Net-BF+SCoDe'))
            if 'superpoint' in extractors:
                pipelines.append((extractors['superpoint'], overlap_estimator, matchers['bf'], 'SuperPoint-BF+SCoDe'))

        # Initialize end-to-end matchers (both with and without SCoDe)
        end2end_matchers = {}

        # SE2-LoFTR matcher initialization (baseline)
        if 'se2_loftr' in args.end2end_matchers and SE2_LOFTR_AVAILABLE:
            try:
                se2_loftr_config = {
                    'weights': 'se2_loftr/4rot-big.ckpt',
                    'match_threshold': 0.2,
                    'max_keypoints': 2048,
                }
                se2_loftr_model = SE2LoFTR(se2_loftr_config, weights_path)
                se2_loftr_model = se2_loftr_model.to(device).eval()
                end2end_matchers['SE2-LoFTR'] = EndToEndMatcher('se2_loftr', se2_loftr_model)
                
                # Add SE2-LoFTR with SCoDe if available
                if overlap_estimator is not None:
                    end2end_matchers['SE2-LoFTR+SCoDe'] = EndToEndMatcher('se2_loftr_scode', se2_loftr_model, overlap_estimator)
                
                print("SE2-LoFTR matcher initialized successfully")
            except Exception as e:
                print(f"Error loading SE2-LoFTR matcher: {e}")

        # Load image pairs
        if args.pairs_file and os.path.exists(args.pairs_file):
            image_pairs = sample_image_pairs_from_file(args.dataset, args.pairs_file, args.num_images)
        else:
            print("Error: Pairs file not found.")
            return

        # Print evaluation information including end-to-end matchers
        print(f"Selected {len(image_pairs)} image pairs for evaluation.")
        print(f"Evaluating {len(pipelines)} traditional pipelines: {[p[3] for p in pipelines]}")
        print(f"Evaluating {len(end2end_matchers)} end-to-end matchers: {list(end2end_matchers.keys())}")

        if args.visualize:
            print(f"Visualization enabled for angles: {args.vis_angles}")

        # Generate angles
        angles = list(range(0, 180, args.angle_step))

        # Evaluate rotation invariance with visualization option
        results = evaluate_rotation_invariance_pipeline_with_end2end(
            image_pairs=image_pairs,
            pipelines=pipelines,
            end2end_matchers=end2end_matchers,
            angles=angles,
            threshold=3.0,
            image_size=image_size,
            visualize=args.visualize,
            vis_angles=args.vis_angles,
            output_dir=args.output_dir
        )

        # Save results to disk
        save_evaluation_results(results, args.dataset_name, args.output_dir)

    # Plot results (whether newly computed or loaded from disk)
    plot_pipeline_results(results, args.dataset_name, args.output_dir, args.show_raw_points)

if __name__ == "__main__":
    main()