from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog


dataset_root = "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches"

# /apps/local/shared/CV703/datasets/iSAID/iSAID_patches/train
# sftp://10.127.30.55/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/train/instancesonly_filtered_train.json

register_coco_instances("isaid_train", {}, f"{dataset_root}/train/instancesonly_filtered_train.json", f"{dataset_root}/train/images")
register_coco_instances("isaid_val", {}, f"{dataset_root}/val/instancesonly_filtered_val.json", f"{dataset_root}/val/images")

print("Number of training images:", len(DatasetCatalog.get("isaid_train")))
print("Number of validation images:", len(DatasetCatalog.get("isaid_val")))

# MetadataCatalog.get("isaid_train").thing_classes = ["Small_Vehicle", "Large_Vehicle", "plane", "storage_tank", "ship", "Swimming_pool", "Harbor", "tennis_court", "Ground_Track_Field", "Soccer_ball_field", "baseball_diamond", "Bridge", "basketball_court", "Roundabout", "Helicopter"]
# "Small_Vehicle", "Large_Vehicle", "plane", "storage_tank", "ship", "Swimming_pool", "Harbor", "tennis_court", "Ground_Track_Field", "Soccer_ball_field", "baseball_diamond", "Bridge", "basketball_court", "Roundabout", "Helicopter"

# MetadataCatalog.get("isaid_train").set(thing_classes=["airplane", "airport", "baseball_diamond", "basketball_court", "bridge", "container_crane", "ground_track_field", "harbor", "helicopter", "helipad", "large_vehicle", "plane", "roundabout", "ship", "small_vehicle"])
# MetadataCatalog.get("isaid_val").set(thing_classes=["airplane", "airport", "baseball_diamond", "basketball_court", "bridge", "container_crane", "ground_track_field", "harbor", "helicopter", "helipad", "large_vehicle", "plane", "roundabout", "ship", "small_vehicle"])

print("Registered dataset 'isaid_train' with", len(DatasetCatalog.get("isaid_train")), "images")
print("Registered dataset 'isaid_val' with", len(DatasetCatalog.get("isaid_val")), "images")

