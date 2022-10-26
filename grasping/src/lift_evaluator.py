
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import mdmm
import meshio
from functools import partial
from pyquaternion import Quaternion
from hydra.utils import to_absolute_path
import yaml

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src'))

import dflex as df
from dflex import sim
from dflex.model import Mesh
from dflex.tests import test_util

df.config.no_grad = True

class LiftEvaluator():
    def __init__(self, lift_config):
        self.inds = [0,1,2,3,4,5,6, 7,8,9,10, 11,12,13,14, 15,16,17,18, 19,20,21,22]

        self.device = lift_config.device
        self.sim_dt = 5e-6
        #self.sim_dt = 1e-5

    def build_model(self, obj_config, with_viewer_mesh=False):
        builder = df.ModelBuilder()

        test_util.urdf_load(
            builder, 
            #to_absolute_path("dflex/tests/assets/allegro_hand_description/allegro_hand_description_right.urdf"),
            "/home/dylanturpin/repos/ros_thermal_grasp/urdf/allegro.urdf",
            df.transform((0.0, 0.0, 0.0), df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi*0.5)), 
            floating=True,
            limit_ke=0.0,#1.e+3,
            limit_kd=0.0)#1.e+2)

        obj = test_util.build_rigid(builder)
        if obj_config.primitive_type == "sphere":
            builder.add_shape_sphere(
                body=obj,
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                radius=obj_config.radius,
                ke=100000.0,
                kd=1000.0,
                kf=1000.0,
                mu=0.5)
        elif obj_config.primitive_type == "box":
            builder.add_shape_box(
                body=obj,
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                hx=obj_config.hx,
                hy=obj_config.hy,
                hz=obj_config.hz,
                ke=100000.0,
                kd=1000.0,
                kf=1000.0,
                mu=0.5
            )
        elif obj_config.primitive_type == "mesh_sdf":
            rescale = obj_config.rescale
            if with_viewer_mesh:
                #mesh_pv = pv.read(to_absolute_path(obj_config.mesh_path))
                mesh = meshio.read(to_absolute_path(obj_config.mesh_path))
                faces = mesh.cells_dict["triangle"].flatten()
                vertices = mesh.points
                #faces = test_util.pyvistaToTrimeshFaces(np.array(mesh_pv.faces)).flatten()
                #mesh_norm_factor = 1.0 / 5.09 # get mesh bounding box side legnth to 1
                mesh_norm_factor = 10.0
                builder.add_shape_mesh(
                    body=obj,
                    pos=(0.0, 0.0, 0.0),
                    rot=(0.0, 0.0, 0.0, 1.0),
                    mesh=Mesh(vertices,faces),
                    scale=(mesh_norm_factor*rescale,mesh_norm_factor*rescale,mesh_norm_factor*rescale),
                    ke=100000.0,
                    kd=1000.0,
                    kf=1000.0,
                    mu=0.5
                )

            sdf_data = np.load(to_absolute_path(obj_config.sdf_path), allow_pickle=True).item()
            sdf = sdf_data["sdf"]
            pos = sdf_data["pos"]
            scale = sdf_data["scale"]

            #import matplotlib.pyplot as plt
            #plt.imshow(sdf.reshape((256,256,256),order='F')[:,128,:]*rescale)

            builder.add_shape_sdf(
                body=obj,
                pos=(pos[0]*rescale, pos[1]*rescale, pos[2]*rescale),
                rot=(0.0, 0.0, 0.0, 1.0),
                sdf=torch.tensor(sdf, dtype=torch.float32, device=self.device)*rescale,
                scale=(scale[0]*rescale, scale[1]*rescale, scale[2]*rescale),
                ke=100000.0,
                kd=1000.0,
                kf=1000.0,
                mu=0.5,
            )


        builder.joint_target = np.copy(builder.joint_q)
        builder.joint_target_ke = 0.0*np.array(builder.joint_target_ke)
        builder.joint_target_kd = 0.0*np.array(builder.joint_target_kd)
        builder.joint_limit_ke = 0.0*np.array(builder.joint_limit_ke)
        builder.joint_limit_kd = 0.0*np.array(builder.joint_limit_ke)

        device = "cuda"
        self.model = builder.finalize(adapter=device)
        self.model.ground = False
        self.model.enable_tri_collisions = False
        self.model.gravity = torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32, device=device)
        state = self.model.state()
        self.model.collide(state)

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.box_contact_inds = torch.where(self.model.contact_body1 == 26)[0]
        self.self_contact_inds = torch.where(self.model.contact_body1 != 26)[0]
        self.builder = builder

    def run(self, hand_q):

        state = self.model.state()
        state.joint_q[self.inds] = hand_q

        m_per_dir = 0.05 # shake 5cm in each direction
        #v = 8.0 # shake at 8m/s
        a = 5.0 # accelerate at 8m/s^2
        t_per_step = (m_per_dir / a)**(1/2)
        steps_per_dir = int(t_per_step/ self.sim_dt)
        #steps_per_dir = int((m_per_dir / a) // self.sim_dt)
        dirs = torch.tensor([
                             [ 1.0, 0.0, 0.0],
                             [-1.0, 0.0, 0.0],
                             [-1.0, 0.0, 0.0],
                             [ 1.0, 0.0, 0.0],
                             [ 0.0, 1.0, 0.0],
                             [ 0.0,-1.0, 0.0],
                             [ 0.0,-1.0, 0.0],
                             [ 0.0, 1.0, 0.0],
                             [ 0.0, 0.0, 1.0],
                             [ 0.0, 0.0,-1.0],
                             [ 0.0, 0.0,-1.0],
                             [ 0.0, 0.0, 1.0],
                             ], device=self.device)
        n_dirs = dirs.shape[0]
        n = 1 + n_dirs * steps_per_dir


        motion_path_q = torch.zeros(n, 3, device=self.device)
        motion_path_q[0,:] = hand_q[0:3]
        motion_path_qd = torch.zeros(n, 3, device=self.device)
        motion_path_qdd = torch.zeros(n, 3, device=self.device)
        count = 1
        for i in range(n_dirs):
            for j in range(steps_per_dir):
                motion_path_qdd[count, :] = dirs[i,:] * a
                motion_path_qd[count,:] = motion_path_qd[count-1, :] + self.sim_dt * motion_path_qdd[count, :]
                motion_path_q[count,:] = motion_path_q[count-1, :] + self.sim_dt * motion_path_qd[count, :]
                count += 1

        #import matplotlib.pyplot as plt
        #plt.plot(range(n),motion_path_q[:,0].cpu())
        #plt.plot(range(n),motion_path_q[:,1].cpu())
        #plt.plot(range(n),motion_path_q[:,2].cpu())
        record_every = 100
        history = {
            'joint_q': torch.zeros((n // record_every + 1, self.model.joint_q.shape[0]))
        }

        for i in range(n):
            state.joint_q[self.inds] = hand_q
            state.joint_q[0:3] = motion_path_q[i,:]
            state.joint_qd[self.inds] = 0.0
            state.joint_qd[0:3] = motion_path_qd[i,:]

            state = self.integrator.forward(
                self.model, state, self.sim_dt,
                update_mass_matrix=True)
            if i % record_every == 0:
                print(f"{i} / {n}")
                print(state.joint_q)
                history['joint_q'][i // record_every,:] = state.joint_q.detach().cpu()
            if torch.isnan(state.joint_q).any() or (state.joint_q > 10.0).any():
                break

        target = torch.zeros_like(state.joint_q[-6:])
        metric = ((state.joint_q[-6:] - target)**2).sum()
        results = dict(metric=metric, history=history)
        return results