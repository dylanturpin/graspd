
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import mdmm
import pyvista as pv
from pyquaternion import Quaternion

import os
import sys

from dflex import sim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dflex as df
from model import Mesh
import test_util


def get_Js(model,state):
    Js = []
    for i in range(model.articulation_count):
        J_start = model.articulation_J_start[i]
        J_end = model.articulation_J_start[i+1] if i < (model.articulation_count-1) else state.J.shape[0]

        first_joint = model.articulation_joint_start[i]
        last_joint = model.articulation_joint_start[i+1]

        first_coord = model.joint_q_start[first_joint]
        last_coord = model.joint_q_start[last_joint]

        first_dof = model.joint_qd_start[first_joint]
        last_dof = model.joint_qd_start[last_joint]

        joint_count = last_joint-first_joint
        dof_count = last_dof-first_dof
        coord_count = last_coord-first_coord

        J = state.J[J_start:J_end].reshape(6*joint_count,dof_count)
        Js.append(J)
    return Js

def build_rigid(builder, x=True,y=True,z=True,rpy=True):
    builder.add_articulation()
    rigid = -1
    if rpy:
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0, 0.0, 0.0), df.quat_identity()),
            axis=(1.0, 0.0, 0.0),
            type=df.JOINT_REVOLUTE,
            armature=0.0)
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0, 0.0, 0.0), df.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            type=df.JOINT_REVOLUTE,
            armature=0.0)
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0, 0.0, 0.0), df.quat_identity()),
            axis=(0.0, 0.0, 1.0),
            type=df.JOINT_REVOLUTE,
            armature=0.0)
    if x:
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0, 0.0, 0.0), df.quat_identity()),
            axis=(1.0, 0.0, 0.0),
            type=df.JOINT_PRISMATIC,
            armature=0.0)
    if y:
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0, 0.0, 0.0), df.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            type=df.JOINT_PRISMATIC,
            armature=0.0)
    if z:
        rigid = builder.add_link(
            parent=rigid,
            X_pj=df.transform((0.0, 0.0, 0.0), df.quat_identity()),
            axis=(0.0, 0.0, 1.0),
            type=df.JOINT_PRISMATIC,
            armature=0.0)
    return rigid


builder = df.ModelBuilder()

test_util.urdf_load(
    builder, 
    #"assets/franka_description/robots/franka_panda.urdf", 
    "dflex/tests/assets/allegro_hand_description/allegro_hand_description_right.urdf",
    df.transform((0.0, 0.0, 0.0), df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi*0.5)), 
    floating=True,
    limit_ke=0.0,#1.e+3,
    limit_kd=0.0)#1.e+2)

#builder.joint_q = [-2.5747e-03, -7.4301e-02,  7.9993e-02, -6.1178e-01, -5.9025e-01,
         #4.3604e-01,  2.9529e-01, -1.0137e-01,  5.8423e-01,  6.7335e-01,
#quat = [0.0,0.0,0.0,1.0]
#quat = builder.joint_q[3:7]
torch.random.manual_seed(1)
np.random.seed(0)
quat = Quaternion.random()
pos = -0.1*quat.rotate(torch.tensor([1.0,0.0,0.0]))

builder.joint_q = [pos[0], pos[1], pos[2], quat[1], quat[2],
         quat[3],  quat[0], -1.0137e-01,  5.8423e-01,  6.7335e-01,
         6.5354e-01, -9.9509e-07,  7.0700e-01,  7.6750e-01,  6.9550e-01,
        -6.9293e-07,  7.0700e-01,  7.6750e-01,  6.9550e-01,  6.9031e-01,
         5.2765e-01,  5.7043e-01,  7.2461e-01]
#builder.joint_target = [-2.5747e-03, -7.4301e-02,  7.9993e-02, -6.1178e-01, -5.9025e-01,
         #4.3604e-01,  2.9529e-01, -1.0137e-01,  5.8423e-01,  6.7335e-01,
builder.joint_target = [pos[0], pos[1], pos[2], quat[1], quat[2],
         quat[3],  quat[0], -1.0137e-01,  5.8423e-01,  6.7335e-01,
         6.5354e-01, -9.9509e-07,  7.0700e-01,  7.6750e-01,  6.9550e-01,
        -6.9293e-07,  7.0700e-01,  7.6750e-01,  6.9550e-01,  6.9031e-01,
         5.2765e-01,  5.7043e-01,  7.2461e-01]

# set fingers to mid-range of their limits
inds = [0,1,2,3,4,5,6, 7,8,9,10, 11,12,13,14, 15,16,17,18, 19,20,21,22]
for i in range(0,len(builder.joint_q_start)):
    #builder.joint_target_kd[i] = 1e5
    builder.joint_target_ke[i] = 1e8

    if (builder.joint_type[i] == df.JOINT_REVOLUTE):
        dof = builder.joint_q_start[i]
        if dof not in inds: continue
        mid = (builder.joint_limit_lower[dof] + builder.joint_limit_upper[dof])*0.5
        fully_open = builder.joint_limit_lower[dof]
        fully_closed = builder.joint_limit_upper[dof]

        builder.joint_q[dof] = fully_open
        builder.joint_target[dof] = fully_open


box_A = build_rigid(builder)
builder.add_shape_sphere(
    body=box_A,
    pos=(0.0, 0.0, 0.0),
    rot=(0.0, 0.0, 0.0, 1.0),
    radius=0.06,
    ke=100000.0,
    kd=1000.0,
    kf=1000.0,
    mu=0.5)
#builder.add_shape_box(
    #body=box_A,
    #pos=(0.0, 0.0, 0.0),
    #rot=(0.0, 0.0, 0.0, 1.0),
    #hx=0.03,
    #hy=0.03,
    #hz=0.03,
    #ke=100000.0,
    #kd=1000.0,
    #kf=1000.0,
    #mu=0.5
#)
#def pyvistaToTrimeshFaces(cells):
    #faces = []
    #idx = 0
    #while idx < len(cells):
        #curr_cell_count = cells[idx]
        #curr_faces = cells[idx+1:idx+curr_cell_count+1]
        #faces.append(curr_faces)
        #idx += curr_cell_count+1
    #return np.array(faces)
#
# add box mesh and primitive
#box_pv = pv.Box(level=4, quads=False)
#faces = pyvistaToTrimeshFaces(np.array(box_pv.faces)).flatten()
#vertices = box_pv.points * 0.02
#
#builder.add_shape_mesh(
    #body=box_A,
    #pos=(0.0, 0.0, 0.0),
    #rot=(0.0, 0.0, 0.0, 1.0),
    #mesh=Mesh(vertices,faces),
    #scale=(1.0, 1.0, 1.0),
    #density=1000.0,
    #ke=100000.0,
    #kd=1000.0,
    #kf=1000.0,
    #mu=0.5
#)


#builder.joint_limit_lower = -1000.0*np.ones(9)
#builder.joint_limit_upper = 1000.0*np.ones(9)

box_target = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
builder.joint_q[-6:] = box_target
#builder.joint_qd[:] = 0.0
builder.joint_target = np.copy(builder.joint_q)
builder.joint_target_ke = np.array(builder.joint_target_ke)
builder.joint_target_kd = np.array(builder.joint_target_kd)
builder.joint_target_ke[:] = 0.0
builder.joint_target_kd[:] = 0.0

builder.joint_limit_ke = 0.0*np.array(builder.joint_limit_ke)
builder.joint_limit_kd = 0.0*np.array(builder.joint_limit_ke)

device = "cuda"
box_target = torch.tensor(box_target,device=device)

model = builder.finalize(adapter=device)
model.ground = False
model.enable_tri_collisions = False
model.gravity = torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32, device=device)

integrator = df.sim.SemiImplicitIntegrator()
state = model.state()
model.collide(state)
state = model.state()

experiment_name = "target_forces"


sim_dt = 1e-5
sim_time = 0.0

###### OPTIMIZATION
base_joint_q = torch.clone(state.joint_q[inds])
base_joint_q.requires_grad_()
box_contact_inds = torch.where(model.contact_body1 == 26)[0]
self_contact_inds = torch.where(model.contact_body1 != 26)[0]

# get initial force
state = integrator.forward(
    model, state, sim_dt,
    update_mass_matrix=True)
initial_force = state.contact_f_s[box_contact_inds,:].detach()

self_contact_inds = torch.where(model.contact_body1 != 26)[0]
lr = 1e-4

#def joint_limits():
    #l_joint = -torch.minimum(base_joint_q[7:] - model.joint_limit_lower[inds][7:]*0.7, torch.zeros_like(base_joint_q[7:])).sum() - torch.minimum(model.joint_limit_upper[inds][7:]*0.7 - base_joint_q[7:], torch.zeros_like(base_joint_q[7:])).sum()
    #return l_joint

n_vels = 7
task_vels = torch.vstack((torch.eye(3,device=device),-torch.eye(3,device=device),torch.zeros(1,3,device=device)))
task_vels *= 0.01
def task():
    l = 0.0
    for i in range(n_vels):
        state = model.state()
        #state.joint_qd[-1] = -0.01
        state.joint_qd[-3:] = task_vels[i,:]
        state.joint_q[inds] = base_joint_q

        m = 1
        for k in range(m):
            state = integrator.forward(
                model, state, sim_dt,
                update_mass_matrix=True)
        l += 1e5*((state.joint_qd[-6:] - torch.zeros_like(state.joint_qd[-6:]))**2).sum()
    return l

experiment_name = "current"

task_constraint = mdmm.MaxConstraintHard(task, 1e-2, damping=1.0)
#joint_limits_constraint = mdmm.MaxConstraint(joint_limits, 1e-5, damping=1.0)
mdmm_module = mdmm.MDMM([task_constraint])
optimizer = mdmm_module.make_optimizer([base_joint_q],optimizer=torch.optim.Adamax,lr=lr)
#optimizer_state_dict = torch.load(f"outputs/{experiment_name}_optimizer_state_dict.pth")
#optimizer.load_state_dict(optimizer_state_dict)
#optimizer = torch.optim.Adam([base_joint_q],lr=lr)

render = True
if render:
    # set up Usd renderer
    from pxr import Usd
    from dflex.render import UsdRenderer
    stage_name = f"outputs/{experiment_name}.usd"
    stage = Usd.Stage.CreateNew(stage_name)
    renderer = UsdRenderer(model, stage)
    renderer.draw_points = False
    renderer.draw_springs = False
    renderer.draw_shapes = True

n = 40000
plot_every = 100
record_stats = True
stats = {
    'c_drop': np.zeros(n//plot_every),
    'c_balanced_forces': np.zeros(n//plot_every),
    'l_force': np.zeros(n//plot_every),
    'joint_q': np.zeros((n//plot_every, base_joint_q.shape[0]))
}
for i in range(n):
    state = model.state()
    state.joint_q[inds] = base_joint_q

    #state.joint_qd[-1] = -0.01
    state.joint_qd[:] = 0.0

    m = 1
    for k in range(m):
        state = integrator.forward(
            model, state, sim_dt,
            update_mass_matrix=True)

    l_self_collision = 1e-7*(state.contact_f_s[self_contact_inds,:]**2).sum()

    l_joint = 1e-2*((base_joint_q[7:] - 0.5*(model.joint_limit_lower[inds][7:] + model.joint_limit_upper[inds][7:]))**2).sum()

    #initial_force_dir = initial_force / torch.norm(initial_force)
    #force = state.contact_f_s[box_contact_inds,:]
    #force_dir = force / torch.norm(force)
    #l_force_norm = 1e-4*((force_dir - initial_force_dir)**2).sum()

    #initial_force_balance = (initial_force**2).sum(dim=1)
    #initial_force_balance /= torch.norm(initial_force_balance)
    #force = state.contact_f_s[box_contact_inds,:]
    #force_balance = (force**2).sum(dim=1)
    #force_balance = force_balance / torch.norm(force_balance)
    #l_force_norm = 1e-2*((force_balance - initial_force_balance)**2).sum()

    #force = state.contact_f_s[box_contact_inds,:]
    #force_mag = (force**2).sum(dim=1)
    #l_force_norm = 0.0*1e-4 * (torch.softmax(force_mag,dim=0)*force_mag).sum()

    #force = state.contact_f_s[box_contact_inds,:]
    #l_force_norm = 1e-4 * (force**2).sum()

    #forces = (state.contact_f_s[:]**2).sum(dim=1)
    #l_force = (torch.softmax(1e5*forces,dim=0)*forces).sum()

    l = l_self_collision + l_joint

    mdmm_return = mdmm_module(l)

    if i % plot_every == 0:
        print(f"{mdmm_return.fn_values[0]} {l_self_collision} {l_joint} {base_joint_q}")

        if render:
            renderer.update(state, i / plot_every)
        if record_stats:
            #stats['c_balanced_forces'][i // plot_every] = mdmm_return.fn_values[0].detach().cpu()
            stats['c_drop'][i // plot_every] = mdmm_return.fn_values[0].detach().cpu()
            #stats['l_force'][i // plot_every] = l_physics.detach().cpu()
            stats['joint_q'][i // plot_every,:] = base_joint_q.detach().cpu()


    optimizer.zero_grad()
    mdmm_return.value.backward()
    optimizer.step()
    base_joint_q.data[3:7] /= torch.norm(base_joint_q.data[3:7])

if render:
    stage.Save()
    print(f"Saved USD stage at {stage_name}.")
if record_stats:
    np.save(f"outputs/{experiment_name}.npy", stats)

print("done")
#torch.save(optimizer.state_dict(), f"outputs/{experiment_name}_optimizer_state_dict.pth")
# lambda_0 = -0.2019
# slack = 0.0075