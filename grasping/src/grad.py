
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.spatial.transform import Rotation
from functools import partial

import mdmm
from pyquaternion import Quaternion
from hydra.utils import to_absolute_path

from manotorch.manolayer import ManoLayer

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src'))

import dflex as df
from dflex import sim
from dflex.model import Mesh
from dflex.tests import test_util

class GradOpt():
    def __init__(self, cio_config):
        self.config = cio_config
        self.inds = [0,1,2,3,4,5,6]
        self.n_inds = len(self.inds)

        self.device = cio_config.device
        self.sim_dt = 1e-5

        self.n_vels = 3
        #self.task_vels = torch.vstack((torch.eye(3,device=self.device),-torch.eye(3,device=self.device),torch.zeros(1,3,device=self.device)))
        self.task_vels = torch.tensor([[0.0,0.0,0.0],[1.0,1.0,1.0],[-1.0,-1.0,-1.0]],device=self.device)
        #self.task_vels = torch.tensor([[0.0,0.0,0.0]],device=self.device)
        self.task_vels *= 0.01

        self.quat = None
        self.pos = None

        self.pos_ratio=1.0
        self.quat_ratio=1.0
        #self.pos_ratio=1.0
        #self.quat_ratio=1.0

        #self.pos_ratio=1e-2
        #self.quat_ratio=1e-1
        self.ncomps = 44
        self.mano_layer = ManoLayer(
            mano_assets_root=to_absolute_path('grasping/data/assets_mano'),
            use_pca=True, ncomps=self.ncomps, flat_hand_mean=True).to(self.device)
        self.mano_shape = torch.rand(1, 10, device=self.device)

        self.backoff = 0.2

        self.coarse_to_fine = cio_config.coarse_to_fine

        # VARIES 1.0 for shapenet, 2.0 for YCB
        if cio_config.dataset == "YCB":
            self.mano_ratio = 2.0
            self.initial_backoff = 0.2
        elif cio_config.dataset == "shapenet":
            self.mano_ratio = 1.0
            self.initial_backoff = 0.2


    def build_model(self, obj_config, with_viewer_mesh=False):
        builder = df.ModelBuilder()

        # test_util.urdf_load(
        #     builder, 
        #     #to_absolute_path("dflex/tests/assets/allegro_hand_description/allegro_hand_description_right.urdf"),
        #     to_absolute_path("grasping/data/mano/ManoHand.urdf"),
        #     #"/home/dylanturpin/repos/ros_thermal_grasp/urdf/allegro.urdf",
        #     df.transform((0.0, 0.0, 0.0), df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi*0.5)), 
        #     floating=True,
        #     limit_ke=0.0,#1.e+3,
        #     limit_kd=0.0)#1.e+2)
        # rigid = test_util.build_rigid(builder,rpy=False)
        builder.add_articulation()
        rigid = -1
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0,0.0,0.0), df.quat_identity()),
            axis=(1.0, 0.0, 0.0),
            type=df.JOINT_FREE)

        batch_size=1
        # Generate random shape parameters
        # Generate random pose parameters, including 3 values for global axis-angle rotation
        random_pose = torch.zeros(batch_size, self.ncomps+3, device=self.device)
        mano_output = self.mano_layer(random_pose, self.mano_shape)
        mano_vertices = mano_output.verts
        mano_vertices *= self.mano_ratio
        mano_vertices = mano_vertices[0,:,:].cpu().numpy()
        faces = self.mano_layer.th_faces
        faces = faces.cpu().numpy().flatten()
        #faces = test_util.pyvistaToTrimeshFaces(faces.cpu().numpy()).flatten()

        builder.add_shape_mesh(
            body=rigid,
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            mesh=Mesh(mano_vertices,faces),
            scale=(1.0,1.0,1.0),
            ke=100000.0,
            kd=1000.0,
            kf=1000.0,
            mu=0.5
        )

        if obj_config.primitive_type == "sphere":
            obj = test_util.build_rigid(builder)
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
            obj = test_util.build_rigid(builder)
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
            mesh_pv = pv.read(to_absolute_path(obj_config.mesh_path))
            faces = test_util.pyvistaToTrimeshFaces(np.array(mesh_pv.faces)).flatten()
            # VARIES 10.0 for YCB, 1.0 for ShapeNet
            if self.config.dataset == "YCB":
                mesh_norm_factor = 10.0
            elif self.config.dataset == "shapenet":
                mesh_norm_factor = 1.0
            vertices = mesh_pv.points

            com = mesh_pv.center_of_mass()*rescale*mesh_norm_factor
            #obj = test_util.build_rigid(builder, com=mesh_pv.center_of_mass()*rescale*mesh_norm_factor)
            obj = test_util.build_rigid(builder, com=com)
            #obj = test_util.build_rigid(builder)

            sdf_data = np.load(to_absolute_path(obj_config.sdf_path), allow_pickle=True).item()
            sdf = sdf_data["sdf"]
            #pos = sdf_data["pos"]
            #scale = sdf_data["scale"]

            # multiply by 10 then pad by 0.1
            # upscale = 10.0
            if self.config.dataset == "YCB":
                pad = 0.1 # already scaled by mesh_norm_factor
            elif self.config.dataset == "shapenet":
                pad = 0.06 # already scaled by mesh_norm_factor
            min_bounds = vertices.min(axis=0) * mesh_norm_factor - pad
            max_bounds = vertices.max(axis=0) * mesh_norm_factor + pad
            scale = max_bounds - min_bounds
            pos = 0.5*(min_bounds + max_bounds)
            # m_shape = 256

            # grid_x = np.linspace(min_bounds[0],max_bounds[0],num=m_shape)
            # grid_y = np.linspace(min_bounds[1],max_bounds[1],num=m_shape)
            # grid_z = np.linspace(min_bounds[2],max_bounds[2],num=m_shape)
            # grid_x,grid_y,grid_z = np.meshgrid(grid_x,grid_y,grid_z)
            # query_points = np.concatenate((grid_x[:,:,:,None],grid_y[:,:,:,None],grid_z[:,:,:,None]),axis=-1)

            #v, f = igl.read_triangle_mesh(to_absolute_path(obj_config.mesh_path))
            #mesh_pv = mesh_pv.decimate(target_reduction=0.95)
            #faces = test_util.pyvistaToTrimeshFaces(np.array(mesh_pv.faces))
            #vertices = mesh_pv.points
            #mesh_simplifier = pyfqmr.Simplify()
            #mesh_simplifier.setMesh(vertices, faces)
            #mesh_simplifier.simplify_mesh(target_count = 20000, aggressiveness=1, lossless=True, preserve_border=True, verbose=10)
            #vertices, faces, normals = mesh_simplifier.getMesh()

            #faces,vertices = pymeshfix.clean_from_arrays(vertices,faces)
            #meshfix = pymeshfix.MeshFix(vertices,faces)
            #meshfix.repair()
            #meshfix.plot()
            #fixed_mesh = meshfix.mesh
            # mustard: 16k faces and 50k vertices...
            # shapenet: 50k faces and 25k vertices...

            #vertices *= upscale
            #sdf_, _, _ = igl.signed_distance(query_points.reshape(-1,3), vertices, faces, return_normals=False)

            builder.add_shape_mesh(
                body=obj,
                #pos=(0.0, 0.0, 0.0),
                pos=-com,
                rot=(0.0, 0.0, 0.0, 1.0),
                mesh=Mesh(vertices,faces),
                scale=(mesh_norm_factor*rescale,mesh_norm_factor*rescale,mesh_norm_factor*rescale),
                ke=100000.0,
                kd=1000.0,
                kf=1000.0,
                mu=0.5
            )

            #import matplotlib.pyplot as plt
            #plt.imshow(sdf.reshape((256,256,256),order='F')[:,128,:]*rescale)

            builder.add_shape_sdf(
                body=obj,
                pos=(pos[0]*rescale, pos[1]*rescale, pos[2]*rescale)-com,
                #pos=(pos[0]*rescale, pos[1]*rescale, pos[2]*rescale) - com,
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

        self.model = builder.finalize(adapter=self.device)
        self.model.ground = False
        self.model.enable_tri_collisions = False
        self.model.gravity = torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        state = self.model.state()
        self.model.collide(state)

        self.integrator = df.sim.SemiImplicitIntegrator()

        self.box_contact_inds = torch.where(self.model.contact_body1 == 6)[0]
        self.self_contact_inds = torch.where(self.model.contact_body1 !=6)[0]
        self.builder = builder

        # for i in range(0,len(self.model.joint_type)):
        #     if (self.model.joint_type[i] == df.JOINT_REVOLUTE):
        #         dof = self.model.joint_q_start[i]
        #         if dof not in self.inds: continue
        #         mid = (self.model.joint_limit_lower[dof] + self.model.joint_limit_upper[dof])*0.5
        #         fully_open = self.model.joint_limit_lower[dof]
        #         fully_closed = self.model.joint_limit_upper[dof]

        #         # default joints to fully_open
        #         #self.model.joint_q[dof] = fully_open
        #         self.model.joint_q[dof] = fully_open

        #         # set rotational joints of fingers and thumb to mid
        #         if dof in [7,11,15, 19]:
        #             self.model.joint_q[dof] = mid
                
                # start thumb opposing fingers
                #if dof in [19]:
                    #self.model.joint_q[dof] = fully_closed

        if obj_config.primitive_type == "mesh_sdf":
            # keep restarting until we find a good point
            df.config.no_grad = True
            n_attempts = 1000
            for i in range(n_attempts):
                print(f"attempt {i} to find initialize")
                # pick a random point, get the normal, save the starting pos for the hand
                ind = np.random.choice(np.arange(0,mesh_pv.point_normals.shape[0]))
                pos = vertices[ind,:] * mesh_norm_factor * rescale
                n = mesh_pv.point_normals[ind,:]
                R = test_util.rotation_matrix_from_vectors(np.array([0.0,1.0,0.0]), n)
                if np.isnan(R).any():
                    continue
                R = Rotation.from_matrix(R)

                rand_roll = Quaternion._from_axis_angle(np.array([0.0,1.0,0.0]), np.random.rand()*2*np.pi)
                rand_roll = rand_roll.elements[[1,2,3,0]]
                rand_roll = Rotation.from_quat(rand_roll)
                self.quat = (R*rand_roll).as_quat()
                if np.linalg.norm(n) == 0.0:
                    continue
                n /= np.linalg.norm(n)
                self.pos = pos + self.initial_backoff * n

                mano_input = torch.zeros(self.ncomps+3, device=self.device)
                mano_output = self.mano_layer(mano_input[None,:], self.mano_shape)
                mano_vertices = mano_output.verts
                mano_vertices *= self.mano_ratio
                self.model.contact_point0[self.box_contact_inds,:] = mano_vertices[0,:,:].detach()

                state = self.model.state()
                state.joint_q[0:3] = torch.tensor(self.pos,dtype=torch.float,device=self.device)
                state.joint_q[3:7] = torch.tensor(self.quat,dtype=torch.float,device=self.device)

                m = 1
                for k in range(m):
                    state = self.integrator.forward(
                        self.model, state, self.sim_dt,
                        update_mass_matrix=True)
                
                dist = state.contact_world_dist[self.box_contact_inds].min()
                if state.contact_f_s[self.box_contact_inds,:].abs().sum() == 0.0:# and dist <= 0.27 and dist >= 0.25:
                    self.backoff = dist.detach() - 0.01
                    break
            df.config.no_grad = False

    def sample_initial_guess(self):
        state = self.model.state()
        if self.pos is None:
            quat = Quaternion.random()
            scale = self.model.shape_geo_scale[-1][0]
            scale = torch.tensor([2*scale,2*scale,2*scale])
            obj_pos = self.model.shape_transform[-1,:3]
            #backoff = (scale/2).abs().max()
            #backoff = 0.4
            pos = -0.2 * torch.tensor(quat.rotate(torch.tensor([1.0,0.0,0.0])), device=self.device) + obj_pos
            state.joint_q[:3] = torch.tensor(pos,device=self.device,dtype=torch.float)
            state.joint_q[3:7] = torch.tensor(quat.elements[[1,2,3,0]],device=self.device,dtype=torch.float)

            m = 1
            for k in range(m):
                state = self.integrator.forward(
                    self.model, state, self.sim_dt,
                    update_mass_matrix=True)
            
            dist = state.contact_world_dist[self.box_contact_inds].min()
            self.backoff = dist.detach() - 0.01

            state = self.model.state()
            state.joint_q[:3] = torch.tensor(pos,device=self.device,dtype=torch.float)
            state.joint_q[3:7] = torch.tensor(quat.elements[[1,2,3,0]],device=self.device,dtype=torch.float)
        else:
            pos = self.pos
            quat = self.quat
            state.joint_q[:3] = torch.tensor(pos,device=self.device,dtype=torch.float)
            state.joint_q[3:7] = torch.tensor(quat,device=self.device,dtype=torch.float)
        #quat = self.builder.joint_q[3:7]
        #pos = -0.2*quat.rotate(torch.tensor([1.0,0.0,0.0]))
        #pos = torch.tensor([0.0, -0.07, 0.0])
        #pos = self.builder.joint_q[0:3]
        #state.joint_q[:3] = torch.tensor(pos,device=self.device,dtype=torch.float)
        #state.joint_q[3:7] = torch.tensor(quat.elements[[1,2,3,0]],device=self.device,dtype=torch.float)
        #state.joint_q[3:7] = torch.tensor(quat,device=self.device,dtype=torch.float)

        
        hand_q = torch.zeros(7, device=self.device)
        mano_q = torch.zeros(self.ncomps+3, device=self.device)
        #hand_q = torch.randn(7+self.ncomps+3, device=self.device)
        hand_q[:7] = state.joint_q[:7]
        hand_q[0:3] /= self.pos_ratio
        hand_q[3:7] /= self.quat_ratio
        #hand_q[7:] = 0.1*torch.ones_like(hand_q[7:])

        #for i in range(self.n_vels):
            #contact_forces[i,:,-3:] = 1e6 * self.task_vels[i,:] / self.box_contact_inds.numel()

        return (hand_q, mano_q)

    def run(self, initial_guess):
        hand_q, mano_q = initial_guess

        initial_hand_q = torch.clone(hand_q.detach())

        render = True
        
        if render:
            from pxr import Usd
            from dflex.render import UsdRenderer
            stage_name = f"outputs/current.usd"
            stage = Usd.Stage.CreateNew(stage_name)
            renderer = UsdRenderer(self.model, stage)
            renderer.draw_points = False
            renderer.draw_springs = False
            renderer.draw_shapes = True

        ###### OPTIMIZATION
        #n = 50000
        n = 10000
        n_pose_only=0
        level_sets = torch.zeros(n, device=self.device)
        if self.coarse_to_fine:
            level_sets[n_pose_only:n-5000] = torch.linspace(self.backoff.item(), 0.0, n-5000-n_pose_only)
            level_sets[:n_pose_only] = level_sets[n_pose_only]
        alpha_schedule = torch.linspace(0.0001, 0.0, n)
        record_every = 20

        #inds_by_depth = [0,1,2, 3,4,5,6] + list(reversed([7,11,15,19, 8,12,16,20, 9,13,17,21, 10,14,18,22]))
        #inds_by_depth = [0,1,2, 6,5,4,3, 7,11,15,19, 8,12,16,20, 9,13,17,21, 10,14,18,22]
        #inds_by_depth = [0,1,2, 3,4,5,6, ]
        #inds_by_depth = [list(range(7+self.ncomps+3))]
        #inds_by_depth = [[0,1,2], [0,1,2], list(range(self.ncomps+3))[3:]]
        inds_by_depth = [[0,1,2], list(range(self.ncomps+3))]
        #inds_by_depth = [[0,1,2],[7,16,25,34,43],[8,17,26,35,44],[9,18,27,36,45],[10,19,28,37,46],[11,20,29,38,47],[12,21,30,39,48],[13,22,31,40,49],[14,23,32,41,50],[15,24,33,42,51],[52,53,54]]
        #inds_by_depth = [[0,1,2], [3,4,5,6,], [7,11,15,19], [8,12,16,20], [9,13,17,21], [10,14,18,22]]

        n_inner_hand_q = len(inds_by_depth)
        n_warmup = 200

        # stats for hand_q update loop
        stats_l_physics = torch.zeros((n,n_inner_hand_q, self.n_vels), device=self.device)
        stats_l_self_collision = torch.zeros((n,n_inner_hand_q), device=self.device)
        stats_l_joint = torch.zeros((n,n_inner_hand_q), device=self.device)
        stats_c_joint = torch.zeros((n,n_inner_hand_q), device=self.device)
        stats_hand_q_grad_norm = torch.zeros((n,n_inner_hand_q), device=self.device)
        stats_hand_q_delta = torch.zeros((n,n_inner_hand_q), device=self.device)
        stats_hand_q = torch.zeros((n,n_inner_hand_q,hand_q.shape[0]), device=self.device)
        stats_hand_q_exp_avg = torch.zeros((n,n_inner_hand_q,hand_q.shape[0]), device=self.device)
        #stats_hand_q_exp_inf = torch.zeros((n,n_inner_hand_q,hand_q.shape[0]), device=self.device)
        stats_actual_forces = torch.zeros((n,n_inner_hand_q,self.n_vels,6), device=self.device)
        stats_actual_forces_l1 = torch.zeros((n,n_inner_hand_q,self.n_vels,6), device=self.device)

        mano_q.requires_grad_()
        hand_q.requires_grad_()

        lr_hand_q = 1e-3 # 2e-4 worked well
        lr_mano_q = 3e-3 # 3e-3 worked well
        hand_q_optimizer = torch.optim.Adam([
            {"params": [hand_q], "lr": lr_hand_q},
            {"params": [mano_q], "lr": lr_mano_q}])

        load_opt_state = False
        if load_opt_state:
            checkpoint = torch.load("/home/dylanturpin/repos/dflex_clean_explore/grasping/data/checkpoint.pth")
            for param_group_state in checkpoint["hand_q_optimizer_state_dict"]["state"].values():
                param_group_state["exp_avg"].zero_()
            hand_q_optimizer.load_state_dict(checkpoint["hand_q_optimizer_state_dict"])

        history = {
            'joint_q': torch.zeros((n//record_every+1, self.model.joint_q.shape[0])),
            'mano_q': torch.zeros((n//record_every+1, mano_q.shape[0])),
            'joint_angles': torch.zeros((n//record_every+1, 21, 3)),
            'c_task': torch.zeros((n//record_every+1)),
            'l_self_collision': torch.zeros((n//record_every+1)),
            'l_joint': torch.zeros((n//record_every+1)),
            'actual_contact_forces': torch.zeros(n//record_every+1, self.n_vels, self.box_contact_inds.shape[0], 6)
        }

        box_contact_forces = torch.zeros(self.n_vels, self.box_contact_inds.shape[0], 6)
        for i in range(n):
            self.model.contact_point0 = self.model.contact_point0.detach()
            self.model.contact_kd = level_sets[i].item()
            self.model.contact_alpha = alpha_schedule[i].item()
            if i == 0 and render:
                contact_point0_before = torch.clone(self.model.contact_point0)
                mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
                vertices = mano_output.verts
                vertices *= self.mano_ratio
                self.model.contact_point0[self.box_contact_inds,:]= vertices[0,:,:].detach()
                state = self.model.state()
                state.joint_q[self.inds] = hand_q[:self.n_inds]
                state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
                state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]

                m = 1
                for k in range(m):
                    state = self.integrator.forward(
                        self.model, state, self.sim_dt,
                        update_mass_matrix=True)
                self.model.contact_point0 = contact_point0_before

                m=stage.GetObjectAtPath("/root/body_0/mesh_0")
                m.GetAttribute("points").Set(vertices[0,:,:].detach().cpu().numpy(),i)
                renderer.update(state, i)
                stage.Save()

                state = self.model.state()
                state.joint_q[self.inds] = hand_q[:7]
                state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
                state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]
                history['joint_q'][0,:] = state.joint_q.detach().cpu()
                history['joint_angles'][0,:,:] = mano_output.joints[0,:,:].detach().cpu()
                history['mano_q'][0,:] = mano_q.detach().cpu()
                history['c_task'][0] = 0.0

            #self.model.contact_kd = 0.00001
            #self.model.contact_kd = level_sets[i]
            if i == 0:
                for j in range(self.n_vels):
                    contact_point0_before = torch.clone(self.model.contact_point0)
                    mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
                    vertices = mano_output.verts
                    vertices *= self.mano_ratio
                    self.model.contact_point0[self.box_contact_inds,:] = vertices[0,:,:].detach()
                    state = self.model.state()
                    state.joint_q[self.inds] = hand_q[:self.n_inds]
                    state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
                    state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]
                    state.joint_qd[-3:] = self.task_vels[j,:]
                    state.joint_q[-3:] += self.task_vels[j,:]*self.sim_dt
                    m = 1

                    for k in range(m):
                        state = self.integrator.forward(
                            self.model, state, self.sim_dt,
                            update_mass_matrix=True)
                    #l_physics += 1e-3*((state.contact_f_s[self.box_contact_inds,:]+contact_forces[j,:,:].detach())**2).sum()
                    box_contact_forces[j,:,:] = state.contact_f_s[self.box_contact_inds,:].detach()
                    self.model.contact_point0 = contact_point0_before
                history['actual_contact_forces'][0,:,:,:] = box_contact_forces.detach().cpu()
            real_forces = state.contact_f_s[self.box_contact_inds,:].abs().sum()
            #self.model.contact_kd = 0.0


            initial_hand_q_inner = torch.clone(hand_q.data)
            initial_mano_q_inner = torch.clone(mano_q.data)
            for i_inner in range(n_inner_hand_q):
                # warmup learning rate
                #if (i*n_inner_hand_q+ i_inner)<n_warmup:
                if i<n_warmup:
                    param_group_hand_q = hand_q_optimizer.param_groups[0]
                    param_group_mano_q = hand_q_optimizer.param_groups[1]
                    #for param_group in hand_q_optimizer.param_groups:
                    #param_group["lr"] = (i*n_inner_hand_q+i_inner) * (lr_hand_q / n_warmup)
                    param_group_hand_q["lr"] = (i) * (lr_hand_q / n_warmup)
                    param_group_mano_q["lr"] = (i) * (lr_mano_q / n_warmup)
                else:
                    #for param_group in hand_q_optimizer.param_groups:
                    param_group_hand_q = hand_q_optimizer.param_groups[0]
                    param_group_mano_q = hand_q_optimizer.param_groups[1]
                    param_group_hand_q["lr"] = lr_hand_q
                    param_group_mano_q["lr"] = lr_mano_q

                stats_hand_q[i,i_inner,:] = hand_q.detach()
                l_task = 0.0
                l_self_collision = 0.0
                real_forces = 0.0
                #self.model.contact_kd = 0.00001
                #self.model.contact_kd = level_sets[i]
                for j in range(self.n_vels):
                    contact_point0_before = torch.clone(self.model.contact_point0)
                    mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
                    vertices = mano_output.verts
                    vertices *= self.mano_ratio
                    self.model.contact_point0[self.box_contact_inds,:] = vertices[0,:,:]
                    state = self.model.state()
                    state.joint_q[self.inds] = hand_q[:self.n_inds]
                    state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
                    state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]
                    state.joint_qd[-3:] = self.task_vels[j,:]
                    state.joint_q[-3:] += self.task_vels[j,:]*self.sim_dt
                    m = 1

                    for k in range(m):
                        state = self.integrator.forward(
                            self.model, state, self.sim_dt,
                            update_mass_matrix=True)
                    l = 1e5*((state.joint_qd[-6:])**2).sum()
                    real_forces += state.contact_f_s[self.box_contact_inds,:].detach().abs().sum()
                    stats_actual_forces[i,i_inner,j,:] = state.contact_f_s[self.box_contact_inds,:].sum(dim=0).detach()
                    stats_actual_forces_l1[i,i_inner,j,:] = state.contact_f_s[self.box_contact_inds,:].abs().sum(dim=0).detach()
                    stats_l_physics[i,i_inner, j] = l.detach()
                    l_task += l
                    l_self_collision += (state.contact_f_s[self.self_contact_inds,:]**2).sum()
                    #if i == 0:
                        #_physics -= state.contact_f_s[self.box_contact_inds,:].abs().sum()
                    box_contact_forces[j,:,:] = state.contact_f_s[self.box_contact_inds,:].detach()
                    self.model.contact_point0 = contact_point0_before

                # find out sensitivity of contact positions to q
                #l_contacts = state.contact_world_pos.abs().sum()
                #hand_q_optimizer.zero_grad()
                #l_contacts.backward()

                # so... position is huge
                # quat is huge
                # rotational joint of each finger is lower
                # other joints... well earlier ones have more control
                # to get everything to about 1e2 would be...
                # pos_ratio = 1e-2
                # quat_ratio = 1e-1

                # keep finger rotations near 0
                #middle_inds = []
                middle_inds = self.inds[7:]
                #middle_inds = [7,11,15,19]
                #middle_inds = [7,8,9,10, 11,12,13,14, 15,16,17,18, 19,20,21,22]
                #middle_inds = [7,8,9,10, 11,12,13,14, 15,16,17,18]
                #middle_inds = [7,11,15]
                #l_joint = 1e4*((hand_q[middle_inds] - 0.5*(self.model.joint_limit_lower[middle_inds] + self.model.joint_limit_upper[middle_inds]))**2).sum()
                #l_joint = 1e5*(self.anatomy_metric.compute_loss(mano_output.full_poses,mano_output.joints)-0.0)**2
                l_joint = 1e5*torch.norm(hand_q[middle_inds],dim=0)

                #self.model.contact_kd = 0.0
                stats_l_joint[i,i_inner] = l_joint.detach()
                stats_l_self_collision[i,i_inner] = l_self_collision.detach()
                l = l_task + l_self_collision + l_joint
                #mdmm_return = mdmm_module_hand_q(l)
                #stats_c_joint[i,i_inner] = mdmm_return.fn_values[0].detach()
                stats_c_joint[i,i_inner] = 0.0
                hand_q_optimizer.zero_grad()
                l.backward()
                #mdmm_return.value.backward()
                #if i <= 2500:
                # if i <= 5:
                #     hand_q.grad.data[7:] = 0.0
                stats_hand_q_grad_norm[i,i_inner] = torch.norm(hand_q.grad).detach()
                #over_limit = (hand_q >= self.model.joint_limit_upper[self.inds])
                #under_limit = (hand_q <= self.model.joint_limit_lower[self.inds])
                # limit_upper = torch.ones_like(hand_q[7:])
                # limit_lower = 0.0*torch.ones_like(hand_q[7:])
                # over_limit = (hand_q[7:] >= limit_upper)
                # under_limit = (hand_q[7:] <= limit_lower)
                # neg_grad = hand_q.grad[7:] < 0.0
                # pos_grad = hand_q.grad[7:] > 0.0
                # hand_q.grad.data[7:][(over_limit & neg_grad)] = 0.0
                # hand_q.grad.data[7:][(under_limit & pos_grad)] = 0.0

                hand_q_before = torch.clone(hand_q.detach())
                hand_q_optimizer.step()

                # first step pos
                #inds_by_depth = [0,1,2, 3,4,5,6, 7,11,15,19, 8,12,16,20, 9,13,17,21, 10,14,18,22]
                #i_depth = inds_by_depth[-(i_inner+1)]
                if i_inner == 0:
                    i_depth = inds_by_depth[i_inner]
                    initial_hand_q_inner.data[i_depth] = hand_q.data[i_depth]
                else:
                    i_depth = inds_by_depth[i_inner]
                    initial_mano_q_inner.data[i_depth] = mano_q.data[i_depth]
                hand_q.data[:] = initial_hand_q_inner.data
                mano_q.data[:] = initial_mano_q_inner.data
                # if i_inner == 0:
                #     hand_q.data[3:] = initial_hand_q_inner[3:]
                #     initial_hand_q_inner[:3] = hand_q.data[:3]
                # elif i_inner == 1:
                #     hand_q.data[:3] = initial_hand_q_inner[:3]
                #     hand_q.data[7:] = initial_hand_q_inner[7:]
                #     initial_hand_q_inner[3:7] = hand_q.data[3:7]
                # else:
                #     hand_q.data[:7] = initial_hand_q_inner[:7]

                if i < n_pose_only:
                    hand_q.data[7:] = initial_hand_q[7:]
                
                stats_hand_q_delta[i,i_inner] = torch.norm(hand_q_before - hand_q).detach()
                state_dict = hand_q_optimizer.state_dict()
                stats_hand_q_exp_avg[i,i_inner,:] = state_dict["state"][0]["exp_avg"]
                #stats_hand_q_exp_inf[i,i_inner,:] = state_dict["state"][0]["exp_inf"]

                hand_q.data[3:7] /= torch.norm(hand_q.data[3:7])
                hand_q.data[3:7] /= self.quat_ratio

                #print(f"{i_inner} {mdmm_return.fn_values[0]} {l_physics} {l_self_collision} {l_joint}")

            # plot!
            # plot!
            if i == 99000:
                print("break here")
            show_plots = False
            if show_plots:
                fig, axes = plt.subplots(10)
                axes[0].plot(np.arange((i+1)*n_inner_contact_forces), stats_l_physics_[:(i+1),:,0].flatten().cpu(), label="physics0", alpha=0.7)
                axes[0].plot(np.arange((i+1)*n_inner_contact_forces), stats_l_physics_[:(i+1),:,1].flatten().cpu(), label="physics1", alpha=0.7)
                #axes[0].plot(np.arange((i+1)*n_inner_contact_forces), stats_l_physics_[:(i+1),:,2].flatten().cpu(), label="physics2", alpha=0.7)
                axes[0].legend()
                axes[1].plot(np.arange((i+1)*n_inner_contact_forces), stats_c_task[:(i+1),:].flatten().cpu(), label="task")
                axes[1].legend()
                axes[2].plot(np.arange((i+1)*n_inner_hand_q), stats_l_physics[:(i+1),:,0].flatten().cpu(), label="physics0", alpha=0.7)
                axes[2].plot(np.arange((i+1)*n_inner_hand_q), stats_l_physics[:(i+1),:,1].flatten().cpu(), label="physics1", alpha=0.7)
                #axes[2].plot(np.arange((i+1)*n_inner_hand_q), stats_l_physics[:(i+1),:,2].flatten().cpu(), label="physics2", alpha=0.7)
                axes[2].legend()
                axes[3].plot(np.arange((i+1)*n_inner_hand_q), stats_l_self_collision[:(i+1),:].flatten().cpu(), label="l_self_collision")
                axes[3].legend()
                axes[4].plot(np.arange((i+1)*n_inner_hand_q), stats_l_joint[:(i+1),:].flatten().cpu(), label="l_joint")
                axes[4].legend()
                axes[5].plot(np.arange((i+1)*n_inner_hand_q), stats_c_joint[:(i+1),:].flatten().cpu(), label="c_joint")
                axes[5].legend()
                axes[6].plot(np.arange((i+1)*n_inner_contact_forces), stats_contact_forces_grad_norm[:(i+1),:].flatten().cpu(), label="contact_forces_grad_norm")
                axes[6].legend()
                axes[7].plot(np.arange((i+1)*n_inner_contact_forces), stats_contact_forces_delta[:(i+1),:].flatten().cpu(), label="norm_contact_forces_delta")
                axes[7].legend()
                axes[8].plot(np.arange((i+1)*n_inner_hand_q), stats_hand_q_grad_norm[:(i+1),:].flatten().cpu(), label="hand_q_grad_norm")
                axes[8].legend()
                axes[9].plot(np.arange((i+1)*n_inner_hand_q), stats_hand_q_delta[:(i+1),:].flatten().cpu(), label="norm_hand_q_delta")
                axes[9].legend()

                fig2, axes2 = plt.subplots(12,2)
                for l in range(hand_q.shape[0]):
                    axes2[l % 12, l // 12].plot(np.arange((i+1)*n_inner_hand_q), stats_hand_q[:(i+1),:,l].flatten().cpu(), label=str(l))
                    if l>6:
                        axes2[l % 12, l // 12].axhline(y=self.model.joint_limit_lower[l], color='r', linestyle='--', alpha=0.1)
                        axes2[l % 12, l // 12].axhline(y=self.model.joint_limit_upper[l], color='r', linestyle='--', alpha=0.1)
                    axes2[l % 12, l // 12].legend()

                fig3, axes3 = plt.subplots(12,2)
                for l in range(hand_q.shape[0]):
                    axes3[l % 12, l // 12].plot(np.arange((i+1)*n_inner_hand_q), stats_hand_q_exp_avg[:(i+1),:,l].flatten().cpu(), label=f"hand_q_exp_avg{l}")
                    axes3[l % 12, l // 12].legend()

                #fig4, axes4 = plt.subplots(12,2)
                #for l in range(hand_q.shape[0]):
                    #axes4[l % 12, l // 12].plot(np.arange((i+1)*n_inner_hand_q), stats_hand_q_exp_inf[:(i+1),:,l].flatten().cpu(), label=f"hand_q_exp_inf{l}")
                    #axes4[l % 12, l // 12].legend()

                fig5, axes5 = plt.subplots(4,6)
                for l in range(6):
                    for m in range(self.n_vels):
                        axes5[0, l].plot(np.arange((i+1)*n_inner_contact_forces), stats_contact_forces[:(i+1),:,m,l].flatten().cpu(), label=f"contact_forces sim{m} dim{l}", alpha=0.7)
                    axes5[0,l].legend()

                    for m in range(self.n_vels):
                        axes5[1, l].plot(np.arange((i+1)*n_inner_contact_forces), stats_contact_forces_l1[:(i+1),:,m,l].flatten().cpu(), label=f"contact_forces_l1 sim{m} dim{l}", alpha=0.7)
                    axes5[1,l].legend()

                    for m in range(self.n_vels):
                        #alpha= 1.0 if m == 0 else 0.1
                        alpha=0.7
                        axes5[2,l].plot(np.arange((i+1)*n_inner_hand_q), stats_actual_forces[:(i+1),:,m,l].flatten().cpu(), label=f"actual_forces sim{m} dim{l}", alpha=alpha)
                    axes5[2,l].legend()

                    for m in range(self.n_vels):
                        #alpha= 1.0 if m == 0 else 0.01
                        alpha=0.7
                        axes5[3,l].plot(np.arange((i+1)*n_inner_hand_q), stats_actual_forces_l1[:(i+1),:,m,l].flatten().cpu(), label=f"actual_forces_l1 sim{m} dim{l}", alpha=alpha)
                    axes5[3,l].legend()
                fig.tight_layout()
                fig2.tight_layout()
                fig3.tight_layout()
                fig4.tight_layout()
                fig5.tight_layout()
                fig.show()
                fig2.show()
                fig3.show()
                fig4.show()
                fig5.show()


            if i % record_every == 0:
                mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
                if render:
                    vertices = mano_output.verts
                    vertices *= self.mano_ratio

                    m=stage.GetObjectAtPath("/root/body_0/mesh_0")
                    #vertices = np.array(m.GetAttribute("points").Get(i))*0.9
                    #m.GetAttribute("points").Set(vertices,i+1.0)
                    m.GetAttribute("points").Set(vertices[0,:,:].detach().cpu().numpy(),i+1)
                    renderer.update(state, i+1)
                    stage.Save()

                print(f"{i}/{n} task:{l_task} l_joint:{l_joint} {hand_q} {mano_q} real_forces_l1: {real_forces}")
                #print(f"{i}/{n} {mdmm_return.fn_values[0]} {l_self_collision} {mdmm_return.fn_values[1]} {l_physics} {hand_q} {state.contact_f_s[self.box_contact_inds,:].abs().sum()} {contact_forces.abs().sum()}")

                state = self.model.state()
                state.joint_q[self.inds] = hand_q[:7]
                state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
                state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]
                history['joint_q'][i // record_every+1,:] = state.joint_q.detach().cpu()
                history['mano_q'][i // record_every+1,:] = mano_q.detach().cpu()
                history['joint_angles'][i // record_every+1,:,:] = mano_output.joints[0,:,:].detach().cpu()
                history['c_task'][i // record_every+1] = l_task.detach().cpu()
                history['l_self_collision'][i // record_every+1] = l_self_collision.detach().cpu()
                history['l_joint'][i // record_every + 1] = l_joint.detach().cpu()
                history['actual_contact_forces'][i // record_every +1,:,:,:] = box_contact_forces.detach().cpu()
                #history['l_joint'][i // record_every] = mdmm_return.fn_values[1].detach().cpu()

        record_opt_state = False
        if record_opt_state:
            print("recording")
            path="/home/dylanturpin/repos/dflex_clean_explore/grasping/data/checkpoint.pth"
            torch.save({
                "hand_q_optimizer_state_dict": hand_q_optimizer.state_dict(),
                "contact_forces_optimizer_state_dict": contact_forces_optimizer.state_dict(),
                "mdmm_module_state_dict": mdmm_module.state_dict(),
                "mdmm_module_hand_q_state_dict": mdmm_module_hand_q.state_dict()
            },
            "path.pth")




        # record final results
        #print(f"{mdmm_return.fn_values[0]} {l_self_collision} {l_joint} {l_physics}")
        #print(f"{mdmm_return.fn_values[0]} {l_self_collision} {mdmm_return.fn_values[1]} {l_physics}")
        mano_output = self.mano_layer(mano_q[None,:], self.mano_shape)
        state = self.model.state()
        state.joint_q[self.inds] = hand_q
        state.joint_q[self.inds[0:3]] = self.pos_ratio*hand_q[0:3]
        state.joint_q[self.inds[3:7]] = self.quat_ratio * hand_q[3:7]
        results = {
            'final_joint_q': state.joint_q.detach().cpu(),
            'final_mano_q': mano_q.detach().cpu(),
            'final_joint_angles': mano_output.joints[0,:,:].detach().cpu(),
            'final_c_task': l_task.detach().cpu(),
            'final_l_self_collision': l_self_collision.detach().cpu(),
            'final_l_joint': l_joint.detach().cpu(),
            'actual_contact_forces': box_contact_forces.detach().cpu(),
            #'final_l_joint': mdmm_return.fn_values[1].detach().cpu(),
            'history': history
        }
        return results

    def joint_limits(self, hand_q):
        l_joint = -torch.minimum(hand_q[7:] - self.model.joint_limit_lower[self.inds][7:], torch.zeros_like(hand_q[7:])).sum() - torch.minimum(self.model.joint_limit_upper[self.inds][7:] - hand_q[7:], torch.zeros_like(hand_q[7:])).sum()
        #l_joint += (1.0 - torch.norm(hand_q[3:7])).abs()
        return 1e7*l_joint