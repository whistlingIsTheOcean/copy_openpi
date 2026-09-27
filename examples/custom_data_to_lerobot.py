"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
#import tensorflow_datasets as tfds
import tyro
import os 
import numpy as np
import json
from PIL import Image

from flexiv_dataset_to_lerobot.constants import *

from examples.our_real.utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform


REPO_NAME = "real_data_0710_0921trans/libero"  # Name of the output dataset, also used for the Hugging Face Hub
# RAW_DATASET_NAMES = [
#     "libero_10_no_noops",
#     "libero_goal_no_noops",
#     "libero_object_no_noops",
#     "libero_spatial_no_noops",
# ]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main( *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=7,
        features={
            "image": {
                "dtype": "image",
                "shape": (720, 1280, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (720, 1280, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (10,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (10,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    # for raw_dataset_name in RAW_DATASET_NAMES:
    #     raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
    #     for episode in raw_dataset:
    #         for step in episode["steps"].as_numpy_iterator():
    #             dataset.add_frame(
    #                 {
    #                     "image": step["observation"]["image"],
    #                     "wrist_image": step["observation"]["wrist_image"],
    #                     "state": step["observation"]["state"],
    #                     "actions": step["action"],
    #                     "task": step["language_instruction"].decode(),
    #                 }
    #             )
    #         dataset.save_episode()
    dataset_dir="/data/yuxuan/realdata_sampled_20260710"
    dataset_dir=os.path.join(dataset_dir,"train")
    episode_folds=sorted(os.listdir(dataset_dir))
    camera_names =['104122063550', '043322070878']#the former one is master camera with global perspective,while the latter one is slave camera with wrist perspective
    
    for episode in episode_folds:
        demo_path=os.path.join(dataset_dir,episode)
        
        with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                                        meta = json.load(f)
        
        #tcp gripper使用第一个相机（全局主相机）的文件夹
        cam_path=os.path.join(demo_path,f"cam_{camera_names[0]}")
        wrist_cam_path=os.path.join(demo_path,f"cam_{camera_names[1]}")
        tcp_path=os.path.join(cam_path,'tcp')
        gripper_path=os.path.join(cam_path,'gripper_command')
        
        master_frame_ids = [
                                    int(os.path.splitext(x)[0]) 
                                    for x in sorted(os.listdir(os.path.join(demo_path,f"cam_{camera_names[0]}", "color"))) 
                                    if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                                ]
        slave_frame_ids = [
                                int(os.path.splitext(x)[0]) 
                                for x in sorted(os.listdir(os.path.join(demo_path,f"cam_{camera_names[1]}", "color"))) 
                                if int(os.path.splitext(x)[0]) <= meta["finish_time"]
                            ]
        
        demo_len=len(master_frame_ids)
        for cur_frame_idx in range(demo_len):
            master_frame_id=master_frame_ids[cur_frame_idx]
            master_next_frame_id=master_frame_ids[cur_frame_idx+1 if cur_frame_idx+1<demo_len else cur_frame_idx]#the last frame has no descendant,copy itself
            
            # find the wrist(slave) camera frame id correspondant to current master camera frame id 
            slave_frame_ids_np=np.array(slave_frame_ids)
            slave_frame_id=slave_frame_ids[np.argmin(np.abs(slave_frame_ids_np - master_frame_id))]
            
            # load joint angle and gripper width from file
            cur_tcp=np.load(os.path.join(tcp_path,f'{master_frame_id}.npy'))[:7]
            cur_tcp = xyz_rot_transform(cur_tcp, from_rep = "quaternion", to_rep = "rotation_6d")
            cur_gripper=decode_gripper_width(np.load(os.path.join(gripper_path,f'{master_frame_id}.npy'))[0])
            cur_state=np.concatenate([cur_tcp,[cur_gripper]]).astype(np.float32)
            
            tar_tcp=np.load(os.path.join(tcp_path,f'{master_next_frame_id}.npy'))[:7]
            tar_tcp = xyz_rot_transform(tar_tcp, from_rep = "quaternion", to_rep = "rotation_6d")
            tar_gripper=decode_gripper_width(np.load(os.path.join(gripper_path,f'{master_next_frame_id}.npy'))[0])
            tar_state=np.concatenate([tar_tcp,[tar_gripper]]).astype(np.float32)
            
            language_instruction="Pick up the colored cup and move it into the metal cup"
            
            dataset.add_frame(
                    {
                        "image": np.array(Image.open(os.path.join(cam_path, "color", f"{master_frame_id}.png"))),
                        "wrist_image": np.array(Image.open(os.path.join(wrist_cam_path, "color", f"{slave_frame_id}.png"))),
                        "state": cur_state,
                        "actions": tar_state,
                        "task": language_instruction,
                    }
                )
        dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "flexiv"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

#复制自RISE，用于夹爪宽度量纲统一
def decode_gripper_width(gripper_width):
    return gripper_width / 1000. * 0.095

if __name__ == "__main__":
    tyro.cli(main)
