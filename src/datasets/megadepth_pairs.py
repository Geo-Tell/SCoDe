import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
from src.datasets.utils import recover_pair


class MegaDepthPairsDataset(Dataset):
    def __init__(
            self,
            pairs_list_path='assets/train_scenes_all.txt',
            base_path='assets/megadepth',
            train=True,
            pairs_per_scene=1000,
            image_size=[1024, 1024],
            debug=False,
    ):
        """Generate training datasets from preprocessed txt file."""
        # Read all image pairs information
        with open(pairs_list_path, 'r') as f:
            self.total_pairs = [line.split() for line in f.readlines()]

        self.base_path = base_path
        self.train = train
        self.pairs_per_scene = pairs_per_scene
        self.image_size = image_size
        self.debug = debug
        
        
        if self.debug:
            os.makedirs('outputs/temp', exist_ok=True)

        self.dataset = []
        self.total_dataset = []
        self._init_dataset()

    def _is_valid_bbox(self, bbox):
        """Check if bbox is valid"""
        return (len(bbox) >= 4 and 
                bbox[2] > bbox[0] and bbox[3] > bbox[1] and
                bbox[2] - bbox[0] >= 10 and bbox[3] - bbox[1] >= 10)

    def _init_dataset(self):
        """Initialize datasets from preprocessed data."""
        self.total_dataset = []
        
        for data in self.total_pairs:
            bbox1 = np.array(data[1].split(','), dtype=float)
            bbox2 = np.array(data[3].split(','), dtype=float)

            # Validation check for merging
            if not (self._is_valid_bbox(bbox1) and self._is_valid_bbox(bbox2)):
                continue
                
            self.total_dataset.append({
                'image_path1': data[0],
                'overlap1': bbox1,
                'image_path2': data[2],
                'overlap2': bbox2,
                'image_path_n': data[-1],
            })

    def build_dataset(self):
        """Build dataset after every epoch."""
        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)
            
        if len(self.total_dataset) == 0:
            raise ValueError("No valid image pairs in total_dataset")
        
        # Generate valid central match
        valid_data = []
        for data in self.total_dataset:
            central_match = self._generate_central_match(data)
            if central_match is not None:
                data['central_match'] = central_match
                valid_data.append(data)
        
        if len(valid_data) == 0:
            raise ValueError("No valid central matches found")
        
        # Select data
        if self.pairs_per_scene:
            target_size = min(self.pairs_per_scene, len(valid_data))
            selected_ids = np.random.choice(len(valid_data), target_size, replace=False)
            self.dataset = np.array(valid_data)[selected_ids]
        else:
            self.dataset = np.array(valid_data)

        if self.train:
            np.random.shuffle(self.dataset)
        else:
            np.random.set_state(np_random_state)

    def _generate_central_match(self, data):
        """Generate smart central matching points"""
        overlap1, overlap2 = data['overlap1'], data['overlap2']
        
        # Calculate center points
        center1 = [(overlap1[0] + overlap1[2]) / 2, (overlap1[1] + overlap1[3]) / 2]
        center2 = [(overlap2[0] + overlap2[2]) / 2, (overlap2[1] + overlap2[3]) / 2]
        
        # Check if overlap region is large enough
        # min_size = self.image_size[0] * 0.3
        min_size = 5
        if (overlap1[2] - overlap1[0] < min_size or overlap1[3] - overlap1[1] < min_size or
            overlap2[2] - overlap2[0] < min_size or overlap2[3] - overlap2[1] < min_size):
            return None
        
        # Calculate safe sampling range
        crop_margin = self.image_size[0] * 0.25  # 25% margin
        
        safe_bounds1 = [
            overlap1[0] + crop_margin, overlap1[1] + crop_margin,
            overlap1[2] - crop_margin, overlap1[3] - crop_margin
        ]
        safe_bounds2 = [
            overlap2[0] + crop_margin, overlap2[1] + crop_margin,
            overlap2[2] - crop_margin, overlap2[3] - crop_margin
        ]
        
        # Check safe bounds
        if (safe_bounds1[0] >= safe_bounds1[2] or safe_bounds1[1] >= safe_bounds1[3] or
            safe_bounds2[0] >= safe_bounds2[2] or safe_bounds2[1] >= safe_bounds2[3]):
            point2D1, point2D2 = center1, center2
        else:
            if self.train:
                # Random sample in training
                point2D1 = [
                    np.random.uniform(safe_bounds1[0], safe_bounds1[2]),
                    np.random.uniform(safe_bounds1[1], safe_bounds1[3])
                ]
                # Calculate correspondence
                x_ratio = (point2D1[0] - overlap1[0]) / (overlap1[2] - overlap1[0])
                y_ratio = (point2D1[1] - overlap1[1]) / (overlap1[3] - overlap1[1])
                point2D2 = [
                    (overlap2[2] - overlap2[0]) * x_ratio + overlap2[0],
                    (overlap2[3] - overlap2[1]) * y_ratio + overlap2[1]
                ]
            else:
                # Use center point in validation
                point2D1, point2D2 = center1, center2
        
        return np.array([point2D1[1], point2D1[0], point2D2[1], point2D2[0]])

    def _visualize_debug(self, idx, original_data, processed_data):
        """Visualize debug info"""
        if not self.debug:
            return
        
        # Load raw image
        img_path1 = os.path.join(self.base_path, original_data['image_path1'])
        img_path2 = os.path.join(self.base_path, original_data['image_path2'])
        
        orig_img1 = cv2.imread(img_path1)
        orig_img2 = cv2.imread(img_path2)
        
        if orig_img1 is None or orig_img2 is None:
            print(f"Warning: Could not load images for debug visualization")
            return
        
        # Draw original image and bbox
        orig_img1_vis = orig_img1.copy()
        orig_img2_vis = orig_img2.copy()
        
        # Target bbox
        bbox1_orig = original_data['overlap1'].astype(int)
        bbox2_orig = original_data['overlap2'].astype(int)
        
        cv2.rectangle(orig_img1_vis, (bbox1_orig[0], bbox1_orig[1]), 
                     (bbox1_orig[2], bbox1_orig[3]), (0, 255, 0), 3)
        cv2.rectangle(orig_img2_vis, (bbox2_orig[0], bbox2_orig[1]), 
                     (bbox2_orig[2], bbox2_orig[3]), (0, 255, 0), 3)
        
        # Central match point
        if 'central_match' in original_data:
            cm = original_data['central_match']
            cv2.circle(orig_img1_vis, (int(cm[1]), int(cm[0])), 10, (255, 0, 0), -1)
            cv2.circle(orig_img2_vis, (int(cm[3]), int(cm[2])), 10, (255, 0, 0), -1)
        
        # Processed image and bbox
        proc_img1 = (processed_data['image1'].numpy() * 255).astype(np.uint8)
        proc_img2 = (processed_data['image2'].numpy() * 255).astype(np.uint8)
        
        # Convert grayscale to BGR
        if len(proc_img1.shape) == 2:
            proc_img1 = cv2.cvtColor(proc_img1, cv2.COLOR_GRAY2BGR)
        if len(proc_img2.shape) == 2:
            proc_img2 = cv2.cvtColor(proc_img2, cv2.COLOR_GRAY2BGR)
        
        # CHW to HWC
        if len(proc_img1.shape) == 3 and proc_img1.shape[0] == 3:
            proc_img1 = proc_img1.transpose(1, 2, 0)
        if len(proc_img2.shape) == 3 and proc_img2.shape[0] == 3:
            proc_img2 = proc_img2.transpose(1, 2, 0)
        
        proc_img1_vis = proc_img1.copy()
        proc_img2_vis = proc_img2.copy()
        
        # Processed bbox
        bbox1_proc = processed_data['overlap_box1'].numpy().astype(int)
        bbox2_proc = processed_data['overlap_box2'].numpy().astype(int)
        
        cv2.rectangle(proc_img1_vis, (bbox1_proc[0], bbox1_proc[1]), 
                     (bbox1_proc[2], bbox1_proc[3]), (0, 255, 0), 2)
        cv2.rectangle(proc_img2_vis, (bbox2_proc[0], bbox2_proc[1]), 
                     (bbox2_proc[2], bbox2_proc[3]), (0, 255, 0), 2)
        
        # Resize to same shape for comparison
        target_size = 512
        
        # Resize original
        orig_img1_resized = cv2.resize(orig_img1_vis, (target_size, target_size))
        orig_img2_resized = cv2.resize(orig_img2_vis, (target_size, target_size))
        
        # Resize processed
        proc_img1_resized = cv2.resize(proc_img1_vis, (target_size, target_size))
        proc_img2_resized = cv2.resize(proc_img2_vis, (target_size, target_size))
        
        # Create comparison plot
        top_row = np.hstack([orig_img1_resized, orig_img2_resized])
        bottom_row = np.hstack([proc_img1_resized, proc_img2_resized])
        
        # Add labels
        cv2.putText(top_row, "Original Images + Overlap Boxes", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(bottom_row, "Processed Images + Overlap Boxes", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add more info
        cv2.putText(top_row, f"CM: ({int(cm[1])},{int(cm[0])})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(top_row, f"CM: ({int(cm[3])},{int(cm[2])})", (target_size + 10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Concat rows
        comparison = np.vstack([top_row, bottom_row])
        
        # Save image
        save_path = f'outputs/temp/debug_sample_{idx:04d}.jpg'
        cv2.imwrite(save_path, comparison)
        
        # Print debug info
        print(f"Debug Sample {idx}:")
        print(f"  Original bbox1: {bbox1_orig}")
        print(f"  Processed bbox1: {bbox1_proc}")
        print(f"  Original bbox2: {bbox2_orig}")
        print(f"  Processed bbox2: {bbox2_proc}")
        print(f"  Central match: {cm}")
        print(f"  Saved visualization: {save_path}")
        print("-" * 50)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_data = self.dataset[idx]
        
        (
            image1, bbox1, resize_ratio1,
            image2, bbox2, resize_ratio2,
            image_n,
        ) = recover_pair(self.base_path, self.image_size, original_data)

        processed_data = {
            'image1': torch.from_numpy(image1 / 255.0).float(),
            'overlap_box1': torch.from_numpy(bbox1.astype(np.float32)),
            'ratio1': torch.from_numpy(np.asarray(resize_ratio1, np.float32)),
            'image2': torch.from_numpy(image2 / 255.0).float(),
            'overlap_box2': torch.from_numpy(bbox2.astype(np.float32)),
            'ratio2': torch.from_numpy(np.asarray(resize_ratio2, np.float32)),
            'image_n': torch.from_numpy(image_n / 255.0).float(),
            'image_path1': original_data['image_path1'],
            'image_path2': original_data['image_path2'],
            'image_path_n': original_data['image_path_n'],
        }
        
        
        # self._visualize_debug(idx, original_data, processed_data)
        
        return processed_data
