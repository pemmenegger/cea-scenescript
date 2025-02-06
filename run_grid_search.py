import sys
if 'cluster/project/dewolf/pemmenegger/scenescript' not in sys.path:
    sys.path.append('cluster/project/dewolf/pemmenegger/scenescript')

from src.data.point_cloud import PointCloud
from src.networks.scenescript_model import SceneScriptWrapper

ckpt_path = "./weights/scenescript_model_ase.ckpt"
model_wrapper = SceneScriptWrapper.load_from_checkpoint(ckpt_path).cuda()

point_cloud_paths = [
    "../pointclouds/ase/0_semidense_points.csv.gz",
    "../pointclouds/ase/1_semidense_points.csv.gz",
    "../pointclouds/ase/2_semidense_points.csv.gz",
    "../pointclouds/hxe/facade_1.csv.gz",
    "../pointclouds/hxe/facade_2.csv.gz",
    "../pointclouds/hxe/room_B1_ios.csv.gz",
    "../pointclouds/hxe/room_B1.csv.gz",
    "../pointclouds/hxe/room_C1_1.csv.gz",
    "../pointclouds/hxe/room_C1_2.csv.gz",
    "../pointclouds/hxe/room_C1_3.csv.gz",
    "../pointclouds/hxe/room_C1_4.csv.gz",
]

lang_string_paths = [
    "./results/ase_0",
    "./results/ase_1",
    "./results/ase_2",
    "./results/hxe_facade_1",
    "./results/hxe_facade_2",
    "./results/hxe_room_B1_ios",
    "./results/hxe_room_B1",
    "./results/hxe_room_C1_1",
    "./results/hxe_room_C1_2",
    "./results/hxe_room_C1_3",
    "./results/hxe_room_C1_4",
]

print("Running inference on point clouds...")

nucleus_sampling_thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

for i, point_cloud_path in enumerate(point_cloud_paths):
    print(f"Running inference on '{point_cloud_path}' ...")
    point_cloud_obj = PointCloud.load_from_file(point_cloud_path)

    for nucleus_sampling_thresh in nucleus_sampling_thresholds:
        lang_seq = model_wrapper.run_inference(
            point_cloud_obj.points,
            nucleus_sampling_thresh,  # 0.0 is argmax, 1.0 is random sampling
            verbose=True,
        )

        language_string = lang_seq.generate_language_string()
        filename = lang_string_paths[i] + "#" + str(nucleus_sampling_thresh) + ".txt"
        with open(filename, 'w') as file:
            file.write(language_string)

        print(f"{point_cloud_path} with threshold {nucleus_sampling_thresh} done!")

print("All Done!")