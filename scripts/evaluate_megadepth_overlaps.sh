python3 evaluation.py --input_dir /Data/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/MegaDepth_Val_Scales_all.txt --output_dir outputs/MegaDepth_Val_Scales_all --matcher superglue_outdoor --extractor superpoint_aachen  --resize -1 --save --viz
python3 evaluation.py --input_dir /Data/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/MegaDepth_Val_Scales_all.txt --output_dir outputs/MegaDepth_Val_Scales_all --matcher superglue_outdoor --extractor superpoint_aachen  --resize 480 --save --overlaper scode --viz

echo 'without Overlap'
python3 evaluation.py --input_dir /Data/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor superpoint_aachen  --resize -1 --save
python3 evaluation.py --input_dir /Data/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor disk-desc  --resize -1 --save
python3 evaluation.py --input_dir /Data/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor r2d2-desc  --resize -1 --save
python3 evaluation.py --input_dir /Data/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher superglue_disk --extractor disk-desc  --resize -1 --save
python3 evaluation.py --input_dir /Data/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher superglue_outdoor --extractor superpoint_aachen  --resize -1 --save

python3 evaluation.py --input_dir /Data/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/MegaDepth_Test.txt --output_dir outputs/MegaDepth_Test --matcher superglue_outdoor --extractor superpoint_aachen  --resize -1 --save --viz
python3 evaluation.py --input_dir /Data/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/MegaDepth_Test.txt --output_dir outputs/MegaDepth_Test --matcher superglue_outdoor --extractor superpoint_aachen  --resize 480 --save --overlaper scode --viz

echo 'SIFT'
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/MegaDepth_Val_Scales_sampled_3000.txt --output_dir outputs/SIFT --matcher NN --extractor landmark --resize -1 --save
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/MegaDepth_Val_Scales_sampled_3000.txt --output_dir outputs/SIFT --matcher NN --extractor landmark --resize 640 --save --overlaper scode

echo 'with SCoDe'
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_22 --matcher loftr --direct --resize 640 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_22 --matcher NN --extractor d2net-ss  --resize 640 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_22 --matcher NN --extractor context-desc  --resize 640 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_22 --matcher NN --extractor aslfeat-desc  --resize 640 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor superpoint_aachen  --resize 640 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor disk-desc  --resize 640 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher NN --extractor r2d2-desc  --resize 640 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_2 --matcher superglue_disk --extractor disk-desc  --resize 640 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/megadepth_22 --matcher superglue_outdoor --extractor superpoint_aachen  --resize 640 --save --overlaper scode

python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/MegaDepth_Val_Scales_sampled_3000.txt --output_dir outputs/LoFTR --matcher loftr --direct --resize -1 --save
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/MegaDepth_Val_Scales_sampled_3000.txt --output_dir outputs/LoFTR --matcher loftr --direct --resize 640 --save --overlaper scode

echo 'eval_scale_inv'
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/Scale_1024 --matcher loftr --direct --resize 1024 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/Scale_1024 --matcher superglue_outdoor --extractor superpoint_aachen  --resize 1024 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/Scale_640 --matcher loftr --direct --resize 640 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/Scale_640 --matcher superglue_outdoor --extractor superpoint_aachen  --resize 640 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/Scale_480 --matcher loftr --direct --resize 480 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/Scale_480 --matcher superglue_outdoor --extractor superpoint_aachen  --resize 480 --save --overlaper scode

python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/Scale_1024 --matcher NN --extractor landmark --resize 1024 --save --overlaper scode

python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/eval --matcher loftr --direct --resize 1024 --save --overlaper scode --viz
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/eval --matcher NN --extractor d2net-ss --resize 1024 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/eval --matcher NN --extractor landmark --resize 1024 --save --overlaper scode --viz
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/eval --matcher NN --extractor disk-desc --resize 1024 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/eval --matcher NN --extractor r2d2-desc --resize 1024 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/eval --matcher NN --extractor context-desc --resize 1024 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/eval --matcher NN --extractor aslfeat-desc --resize 1024 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/eval --matcher NN --extractor superpoint_aachen --resize 1024 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/eval --matcher superglue_disk --extractor disk-desc --resize 1024 --save --overlaper scode
python3 evaluation.py --input_dir /data/nfs/lhj/DenseMatching/MegaDepth --input_pairs ./dataset/megadepth/assets/megadepth_scale_2.txt --output_dir outputs/eval --matcher superglue_outdoor --extractor superpoint_aachen --resize 1024 --save --overlaper scode --viz
