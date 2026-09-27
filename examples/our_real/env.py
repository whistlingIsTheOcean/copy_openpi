from typing import List, Optional  # noqa: UP035
import numpy as np
import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override


from examples.our_real.eval_agent import Agent ,Agent_slave
from examples.our_real.utils.constants import *
from examples.our_real.utils.transformation import rotation_transform ,rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform

class FlexivRealEnvironment(_environment.Environment):
    """An environment for our Flexiv robot."""

    def __init__(
        self,
        camera_names: List = ['104122063550',
                              '043322070878'],  
        
        discretize_rotation: bool =False
    ) -> None:
        
        self.camera_names=camera_names
        self.discretize_rotation=discretize_rotation
        self.agent_master = Agent(
                robot_ip = "192.168.2.100",
                pc_ip = "192.168.2.35",
                gripper_port = "/dev/ttyUSB0",
                camera_serial = camera_names[0]
            )
        self.agent_slave = Agent_slave(
                camera_serial = camera_names[1]
            )
        if self.discretize_rotation:
            self.last_rot = np.array(self.agent_master.ready_rot_6d, dtype = np.float32)
        self.prev_width=None

    @override
    def reset(self) -> None:
        return
        self.agent_master.set_tcp_pose(self.agent_master.ready_pose,
                                   rotation_rep="quaternion", blocking=True)
        self.prev_width = None

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        
        obs_dict={}
        tcp_7d=self.agent_master.get_tcp_pose().astype(np.float32)
        gripper_width=decode_gripper_width(self.agent_master.get_gripper_width())
        # rotation transformation (4d to 6d)
        action_tcp=xyz_rot_transform(tcp_7d[np.newaxis,...], from_rep = "quaternion", to_rep = "rotation_6d")[0,...]
        obs_dict['observation/state']=np.concatenate((action_tcp,[gripper_width]))#(10,)
        
        
        master_image,_=self.agent_master.get_observation()
        slave_image,_=self.agent_slave.get_observation()
        obs_dict['observation/image']=master_image
        obs_dict['observation/wrist_image']=slave_image
        obs_dict['prompt']="Pick up the colored cup and move it into the metal cup"
        
        return obs_dict
        
        

    @override
    def apply_action(self, action: dict) -> None:
        action=action["actions"]
        action_tcp=action[:-1]
        gripper_width_raw=action[-1]
        
        #action_tcp=xyz_rot_transform(action_tcp_raw[np.newaxis,...],from_rep = "rotation_6d", to_rep = "quaternion")[0,...]
        action_tcp[..., :3] = np.clip(action_tcp[..., :3], SAFE_WORKSPACE_MIN + SAFE_EPS, SAFE_WORKSPACE_MAX - SAFE_EPS)
        step_action = np.concatenate([action_tcp, [gripper_width_raw]], axis = -1)   # (dim,)
        if step_action is None:   # no action in the buffer => no movement.
            assert step_action is not None,'[warn]  action is empty.'            
        print(f'step_action: {step_action}')
        step_tcp = step_action[:-1] 
        step_width = step_action[-1] 
        
        if self.discretize_rotation:
            rot_steps = discretize_rotation(self.last_rot, step_tcp[3:], np.pi / 16)
            self.last_rot = step_tcp[3:]
            for rot in rot_steps:
                step_tcp[3:] = rot
                self.agent_master.set_tcp_pose(
                    step_tcp, 
                    rotation_rep = "rotation_6d", 
                    blocking = True
                )
        else:
            self.agent_master.set_tcp_pose(
                step_tcp,
                rotation_rep = "rotation_6d",
                blocking = True
            )
        
        #send gripper width to gripper,where the width to be sent is normalized
        if step_width <= 0.02:
            step_width = 0.0
        if self.prev_width is None or abs(self.prev_width - step_width) > GRIPPER_THRESHOLD:
            self.agent_master.set_gripper_width(step_width, blocking = True)
            self.prev_width = step_width

def rot_diff(rot1, rot2):
    rot1_mat = rotation_transform(
        rot1,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    rot2_mat = rotation_transform(
        rot2,
        from_rep = "rotation_6d",
        to_rep = "matrix"
    )
    diff = rot1_mat @ rot2_mat.T
    diff = np.diag(diff).sum()
    diff = min(max((diff - 1) / 2.0, -1), 1)
    return np.arccos(diff)

def discretize_rotation(rot_begin, rot_end, rot_step_size = np.pi / 16):
    n_step = int(rot_diff(rot_begin, rot_end) // rot_step_size) + 1
    rot_steps = []
    for i in range(n_step):
        rot_i = rot_begin * (n_step - 1 - i) / n_step + rot_end * (i + 1) / n_step
        rot_steps.append(rot_i)
    return rot_steps

# def unnormalize_action(action):
#     action[..., :3] = (action[..., :3] + 1) / 2.0 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
#     action[..., -1] = (action[..., -1] + 1) / 2.0 * MAX_GRIPPER_WIDTH
#     return action

# def normalize_tcp(self, tcp_list):
#     ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
#     tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
#     tcp_list[:, -1] = tcp_list[:, -1] / MAX_GRIPPER_WIDTH * 2 - 1
#     return tcp_list

#复制自RISE，用于夹爪宽度量纲统一

def decode_gripper_width(gripper_width):
    return gripper_width / 1000. * 0.095