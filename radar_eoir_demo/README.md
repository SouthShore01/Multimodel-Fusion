# radar_eoir_demo

## Run
```bash
pip install -r requirements.txt
python main.py --dataset synthetic
python main.py --dataset autoferry --data-root data/autoferry
python main.py --dataset whut_msfvessel --data-root data/whut_msfvessel
python main.py --dataset mtdsp --data-root data/mtdsp
python main.py --dataset mit_marine_perception --data-root data/mit_marine
```

## Real Dataset Evaluation
- **Autoferry** is recommended first because detections and ground truth are often in a common ownship-fixed NED frame; this adapter is implemented to parse JSON/MAT and map to the common schema.
- **WHUT-MSFVessel** can be used for radar-visible-AIS validation; adapter currently template-only until exact annotation format is mapped.
- **MTDSP** can be used for radar-visible-infrared-AIS validation when mapped into adapter. If only radar echoes are available, TODO is to add CFAR+clustering detector.
- **MIT Marine Perception** can be used for ROS bag radar/video/IR/GPS validation; adapter includes dependency checks for rosbag/rosbags.
- Accuracy is computed from ground truth target IDs if available, otherwise by nearest-ground-truth assignment.

## Outputs
- outputs/association_result.png
- outputs/radar_eoir_matching.png
- outputs/visible_ir_matching.png
- outputs/results_<dataset>.csv
- outputs/matching_summary_<dataset>.json

## Dataset download helper (selective)
```bash
python download_datasets.py --datasets autoferry
python download_datasets.py --datasets autoferry mit_marine_perception --out-dir data
```
If direct URLs are unknown/unavailable, the script creates target folders and prints guidance for manual placement.


Autoferry official dataset page: https://autoferry.github.io/sensor_fusion_dataset/
