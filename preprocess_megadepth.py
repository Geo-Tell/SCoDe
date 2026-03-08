import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, Manager
import multiprocessing as mp
from functools import partial
import psutil
import time
from src.datasets.utils import recover_pair


def get_filename_from_paths(image_path1, image_path2, image_path_n):
    """Generate filename from image paths."""
    dir1 = os.path.basename(os.path.dirname(os.path.dirname(image_path1)))
    name1 = os.path.splitext(os.path.basename(image_path1))[0]
    
    dir2 = os.path.basename(os.path.dirname(os.path.dirname(image_path2)))
    name2 = os.path.splitext(os.path.basename(image_path2))[0]
    
    dir_n = os.path.basename(os.path.dirname(os.path.dirname(image_path_n)))
    name_n = os.path.splitext(os.path.basename(image_path_n))[0]
    
    filename = f"{dir1}_{name1}_{dir2}_{name2}_{dir_n}_{name_n}.pkl"
    return filename


def cleanup_temp_files(output_dir):
    """Clean up temporary files in the output directory."""
    import glob
    temp_files = glob.glob(os.path.join(output_dir, '*.tmp'))
    if temp_files:
        print(f"Cleaning up {len(temp_files)} temporary files...")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Failed to remove {temp_file}: {e}")


def process_single_pair(args):
    """Process a single pair of images."""
    idx, data, base_path, image_size, output_dir, skip_existing = args
    
    try:
        # Parse bbox information
        bbox1 = np.array(data[1].split(','), dtype=float)
        bbox2 = np.array(data[3].split(','), dtype=float)
        
        # Validate bbox
        if len(bbox1) < 4 or len(bbox2) < 4:
            return None, f"Invalid bbox length at index {idx}"
            
        if (bbox1[2] <= bbox1[0] or bbox1[3] <= bbox1[1] or 
            bbox2[2] <= bbox2[0] or bbox2[3] <= bbox2[1]):
            return None, f"Invalid bbox coordinates at index {idx}"
        
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        
        if w1 < 10 or w2 < 10 or h1 < 10 or h2 < 10:
            return None, f"Bbox too small at index {idx}"
        
        # Generate filename from image paths
        item_filename = get_filename_from_paths(data[0], data[2], data[-1])
        item_filepath = os.path.join(output_dir, item_filename)
        
        # Check if file already exists (only if not pre-filtered)
        if skip_existing and os.path.exists(item_filepath):
            return "skipped", None  # Skip already processed files
        
        # Create pair metadata
        pair_metadata = {
            'image_path1': data[0],
            'overlap1': bbox1,
            'image_path2': data[2], 
            'overlap2': bbox2,
            'image_path_n': data[-1],
        }
        
        np.random.seed(idx)
        point2D1 = [
            np.random.randint(int(bbox1[0]), int(bbox1[2])),
            np.random.randint(int(bbox1[1]), int(bbox1[3])),
        ]
        x_ratio = (point2D1[0] - bbox1[0]) / (bbox1[2] - bbox1[0])
        y_ratio = (point2D1[1] - bbox1[1]) / (bbox1[3] - bbox1[1])
        
        point2D2_x = (bbox2[2] - bbox2[0]) * x_ratio + bbox2[0]
        point2D2_y = (bbox2[3] - bbox2[1]) * y_ratio + bbox2[1]
        
        pair_metadata['central_match'] = np.array([point2D1[1], point2D1[0], point2D2_y, point2D2_x])
        
        # Process images using recover_pair
        (image1, bbox1_proc, resize_ratio1, image2, bbox2_proc, resize_ratio2, image_n) = recover_pair(
            base_path, image_size, pair_metadata
        )
        
        # Save processed data as individual file
        processed_item = {
            'image1': image1,
            'bbox1': bbox1_proc,
            'resize_ratio1': resize_ratio1,
            'image2': image2,
            'bbox2': bbox2_proc,
            'resize_ratio2': resize_ratio2,
            'image_n': image_n,
            'image_path1': pair_metadata['image_path1'],
            'image_path2': pair_metadata['image_path2'],
            'image_path_n': pair_metadata['image_path_n'],
            'item_filename': item_filename,
        }
        
        temp_filepath = item_filepath + f'.tmp_{os.getpid()}_{idx}'
        try:
            with open(temp_filepath, 'wb') as f:
                pickle.dump(processed_item, f)
            os.rename(temp_filepath, item_filepath)
        except Exception as e:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            raise e
        
        return "success", None
        
    except Exception as e:
        return None, f"Failed to process pair {idx}: {str(e)}"


def check_existing_files(output_dir, total_pairs):
    """Check which files already exist for resume functionality."""
    existing_files = set()
    if os.path.exists(output_dir):
        for idx, data in enumerate(total_pairs):
            filename = get_filename_from_paths(data[0], data[2], data[-1])
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                existing_files.add(idx)
    return existing_files


def get_system_info():
    """Get system resource information."""
    cpu_count_logical = mp.cpu_count()
    cpu_count_physical = psutil.cpu_count(logical=False)
    memory_total = psutil.virtual_memory().total / (1024**3)  # GB
    memory_available = psutil.virtual_memory().available / (1024**3)  # GB
    
    return {
        'logical_cores': cpu_count_logical,
        'physical_cores': cpu_count_physical,
        'memory_total_gb': memory_total,
        'memory_available_gb': memory_available
    }


def monitor_resources():
    """Monitor current resource usage."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_used_gb = (memory.total - memory.available) / (1024**3)
    memory_percent = memory.percent
    
    return {
        'cpu_percent': cpu_percent,
        'memory_used_gb': memory_used_gb,
        'memory_percent': memory_percent
    }


def preprocess_megadepth_dataset(pairs_list_path, base_path, output_dir, image_size=[640, 640], num_workers=None, resume=True):
    """
    Preprocess the entire MegaDepth dataset and save processed data to disk.
    
    Args:
        pairs_list_path (str): Path to the pairs list file
        base_path (str): Base path to the MegaDepth dataset
        output_dir (str): Directory to save preprocessed data
        image_size (list): Target image size
        num_workers (int): Number of worker processes. If None, use CPU count
        resume (bool): Whether to resume from existing files
    """
    # Get and display system information
    sys_info = get_system_info()
    print("=== System Information ===")
    print(f"Physical CPU cores: {sys_info['physical_cores']}")
    print(f"Logical CPU cores (with hyperthreading): {sys_info['logical_cores']}")
    print(f"Total memory: {sys_info['memory_total_gb']:.1f} GB")
    print(f"Available memory: {sys_info['memory_available_gb']:.1f} GB")
    
    # Create output directory and clean up any existing temp files
    os.makedirs(output_dir, exist_ok=True)
    cleanup_temp_files(output_dir)
    
    # Read all image pairs information
    print("Reading pairs list...")
    with open(pairs_list_path, 'r') as f:
        total_pairs = [line.split() for line in f.readlines()]
    
    print(f"Found {len(total_pairs)} pairs to process")
    
    # Check existing files for resume and filter out already processed pairs
    pending_pairs = []
    skipped_count = 0
    
    if resume:
        print("Checking existing files...")
        existing_files = set()
        if os.path.exists(output_dir):
            existing_pkl_files = set(os.listdir(output_dir))
            existing_pkl_files = {f for f in existing_pkl_files if f.endswith('.pkl')}
        else:
            existing_pkl_files = set()
        
        for idx, data in enumerate(tqdm(total_pairs, desc="Filtering processed files")):
            filename = get_filename_from_paths(data[0], data[2], data[-1])
            if filename in existing_pkl_files:
                skipped_count += 1
            else:
                pending_pairs.append((idx, data))
        
        print(f"Found {skipped_count} already processed files")
        print(f"Remaining {len(pending_pairs)} pairs to process")
    else:
        pending_pairs = [(idx, data) for idx, data in enumerate(total_pairs)]
    
    if len(pending_pairs) == 0:
        print("All files already processed!")
        return
    
    # Set number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count() * 3, 32)
        print(f"\n=== Process Configuration ===")
        print(f"Auto-detected {mp.cpu_count()} logical CPU cores")
        print(f"Using {num_workers} worker processes ({num_workers/mp.cpu_count():.1f}x logical cores)")
    elif num_workers == 1:
        print("Running in single-process mode for debugging")
        processed_count = 0
        failed_count = 0
        
        for idx, data in tqdm(pending_pairs, desc="Processing pairs"):
            result = process_single_pair((idx, data, base_path, image_size, output_dir, False))
            status, error = result
            
            if status == "success":
                processed_count += 1
            else:
                failed_count += 1
                if error:
                    print(f"\n{error}")
        
        print(f"Preprocessing completed. Processed: {processed_count}, Failed: {failed_count}, Skipped: {skipped_count}")
        print(f"Processed data saved to: {output_dir}")
        return
    
    print(f"Using {num_workers} worker processes")
    
    # Prepare arguments for multiprocessing (only pending pairs)
    process_args = [(idx, data, base_path, image_size, output_dir, False) 
                   for idx, data in pending_pairs]
    
    # Process pairs using multiprocessing
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    
    try:
        with Pool(processes=num_workers) as pool:
            results = pool.imap(process_single_pair, process_args, chunksize=10)
            
            with tqdm(total=len(pending_pairs), desc="Processing pairs") as pbar:
                for result in results:
                    status, error = result
                    
                    if status == "success":
                        processed_count += 1
                    else:
                        failed_count += 1
                        if error and failed_count <= 10:
                            print(f"\n{error}")
                    
                    total_processed = processed_count + failed_count
                    if total_processed > 0:
                        elapsed_time = time.time() - start_time
                        processing_rate = total_processed / elapsed_time
                        success_rate = processed_count / total_processed * 100
                        
                        pbar.set_postfix({
                            'Success': processed_count,
                            'Failed': failed_count,
                            'Rate': f"{processing_rate:.1f}/s",
                            'Acc': f"{success_rate:.1f}%"
                        })
                    
                    pbar.update(1)
                        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        pool.terminate()
        pool.join()
        cleanup_temp_files(output_dir)
        return
    except Exception as e:
        print(f"\nMultiprocessing error: {e}")
        cleanup_temp_files(output_dir)
        return
    
    # Final cleanup and statistics
    cleanup_temp_files(output_dir)
    total_time = time.time() - start_time
    
    print(f"\n=== Processing Completed ===")
    print(f"Processed: {processed_count}, Failed: {failed_count}, Skipped: {skipped_count}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average processing rate: {(processed_count + failed_count)/total_time:.2f} pairs/second")
    print(f"Processed data saved to: {output_dir}")


def generate_file_list(input_dir, output_txt_path, file_extension='.pkl', relative_path=False, parent_levels=2):
    """
    遍历指定文件夹下的所有文件，将文件路径保存到txt文件中
    
    Args:
        input_dir (str): 要遍历的输入目录
        output_txt_path (str): 输出txt文件的路径
        file_extension (str): 要筛选的文件扩展名，默认为'.pkl'
        relative_path (bool): 是否保存相对路径，默认为False（绝对路径）
        parent_levels (int): 当使用相对路径时，从文件路径末尾保留的目录级数，默认为2
    """
    import glob
    
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return
    
    print(f"Scanning directory: {input_dir}")
    print(f"Looking for files with extension: {file_extension}")
    
    pattern = os.path.join(input_dir, f"**/*{file_extension}")
    all_files = glob.glob(pattern, recursive=True)
    
    valid_files = [f for f in all_files if not os.path.basename(f).startswith('.tmp') and '.tmp_' not in os.path.basename(f)]
    valid_files.sort()
    
    print(f"Found {len(valid_files)} valid files")
    
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    
    with open(output_txt_path, 'w') as f:
        for file_path in tqdm(valid_files, desc="Writing file paths"):
            if relative_path:
                path_parts = file_path.split(os.sep)
                
                if parent_levels > 0:
                    if len(path_parts) > parent_levels:
                        selected_parts = path_parts[-(parent_levels + 1):]
                        final_path = os.sep.join(selected_parts)
                    else:
                        final_path = file_path
                else:
                    final_path = os.path.basename(file_path)
                
                f.write(f"{final_path}\n")
            else:
                f.write(f"{file_path}\n")
    
    print(f"File list saved to: {output_txt_path}")
    print(f"Total files written: {len(valid_files)}")
    
    if valid_files:
        print("\nSample file paths:")
        for i, file_path in enumerate(valid_files[:5]):
            if relative_path:
                path_parts = file_path.split(os.sep)
                if parent_levels > 0:
                    if len(path_parts) > parent_levels:
                        selected_parts = path_parts[-(parent_levels + 1):]
                        display_path = os.sep.join(selected_parts)
                    else:
                        display_path = file_path
                else:
                    display_path = os.path.basename(file_path)
            else:
                display_path = file_path
            print(f"  {i+1}: {display_path}")
        
        if len(valid_files) > 5:
            print(f"  ... and {len(valid_files) - 5} more files")


def generate_train_valid_lists(preprocessed_dir, output_dir, train_ratio=0.8, random_seed=42):
    """
    从预处理的文件中生成训练和验证文件列表
    
    Args:
        preprocessed_dir (str): 预处理文件所在目录
        output_dir (str): 输出列表文件的目录
        train_ratio (float): 训练集比例，默认0.8
        random_seed (int): 随机种子，默认42
    """
    import glob
    import random
    
    pattern = os.path.join(preprocessed_dir, "*.pkl")
    all_files = glob.glob(pattern)
    all_files = [f for f in all_files if not os.path.basename(f).startswith('.tmp') and '.tmp_' not in os.path.basename(f)]
    all_files.sort()
    
    if not all_files:
        print(f"No pkl files found in {preprocessed_dir}")
        return
    
    print(f"Found {len(all_files)} files")
    
    random.seed(random_seed)
    random.shuffle(all_files)
    
    train_size = int(len(all_files) * train_ratio)
    train_files = all_files[:train_size]
    valid_files = all_files[train_size:]
    
    print(f"Train files: {len(train_files)}")
    print(f"Valid files: {len(valid_files)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_list_path = os.path.join(output_dir, 'train_files.txt')
    with open(train_list_path, 'w') as f:
        for file_path in train_files:
            f.write(f"{file_path}\n")
    
    valid_list_path = os.path.join(output_dir, 'valid_files.txt')
    with open(valid_list_path, 'w') as f:
        for file_path in valid_files:
            f.write(f"{file_path}\n")
    
    print(f"Train list saved to: {train_list_path}")
    print(f"Valid list saved to: {valid_list_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess MegaDepth dataset')
    # parser.add_argument('--pairs_list_path', type=str, default='dataset/megadepth/assets/MegaDepth_from_scale_train_V2.txt',
    parser.add_argument('--pairs_list_path', type=str, default='dataset/megadepth/assets/MegaDepth_from_scale_valid_V2.txt',
                        help='Path to pairs list file')
    parser.add_argument('--base_path', type=str, default='/data/nfs/lhj/DenseMatching/MegaDepth',
                        help='Base path to MegaDepth dataset')
    # parser.add_argument('--output_dir', type=str, default='/data/nfs/lhj/DenseMatching/MegaDepth/covis_box_pairs/train',
    parser.add_argument('--output_dir', type=str, default='/data/nfs/lhj/DenseMatching/MegaDepth/covis_box_pairs/valid',
                        help='Output directory for preprocessed data')
    parser.add_argument('--image_size', type=int, nargs=2, default=[1216, 1216],
                        help='Target image size')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes (default: auto)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from existing files (default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Do not resume from existing files')
    
    parser.add_argument('--generate_list', action='store_true',
                        help='Generate file list from preprocessed directory')
    parser.add_argument('--list_output', type=str, default='preprocessed_files.txt',
                        help='Output file path for generated file list')
    parser.add_argument('--relative_path', action='store_true',
                        help='Save relative paths instead of absolute paths')
    parser.add_argument('--parent_levels', type=int, default=2,
                        help='Number of parent directory levels to include in relative paths (0=filename only)')
    parser.add_argument('--generate_train_valid', action='store_true',
                        help='Generate separate train and validation file lists')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    
    args = parser.parse_args()

    if args.generate_list:
        generate_file_list(
            input_dir=args.output_dir,
            output_txt_path=args.list_output,
            file_extension='.pkl',
            relative_path=args.relative_path,
            parent_levels=args.parent_levels
        )
    elif args.generate_train_valid:
        list_output_dir = os.path.dirname(args.list_output) if os.path.dirname(args.list_output) else '.'
        generate_train_valid_lists(
            preprocessed_dir=args.output_dir,
            output_dir=list_output_dir,
            train_ratio=args.train_ratio
        )
    else:
        preprocess_megadepth_dataset(args.pairs_list_path, args.base_path, args.output_dir, 
                                   args.image_size, args.num_workers, args.resume)
