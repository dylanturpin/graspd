0
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import mdmm
import pyvista as pv
from pyquaternion import Quaternion
from sklearn.mixture import GaussianMixture
import cma
from cma.constraints_handler import BoundTransform

import os
import sys

from dflex import sim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cem import CEM
import dflex as df
from model import Mesh
#from gmm import GaussianMixture
import test_util

df.config.no_grad = True

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

builder.joint_q = [-2.5747e-03, -7.4301e-02,  7.9993e-02, -6.1178e-01, -5.9025e-01,
         4.3604e-01,  2.9529e-01, -1.0137e-01,  5.8423e-01,  6.7335e-01,
#quat = [0.0,0.0,0.0,1.0]
#quat = builder.joint_q[3:7]
#torch.random.manual_seed(1)
#quat = Quaternion()
#pos = -0.5*torch.tensor(quat.rotate([1.0,0.0,0.0]))
#builder.joint_q = [pos[0], pos[1], pos[2], quat[1], quat[2],
         #quat[3],  quat[0], -1.0137e-01,  5.8423e-01,  6.7335e-01,
         6.5354e-01, -9.9509e-07,  7.0700e-01,  7.6750e-01,  6.9550e-01,
        -6.9293e-07,  7.0700e-01,  7.6750e-01,  6.9550e-01,  6.9031e-01,
         5.2765e-01,  5.7043e-01,  7.2461e-01]
builder.joint_target = [-2.5747e-03, -7.4301e-02,  7.9993e-02, -6.1178e-01, -5.9025e-01,
         4.3604e-01,  2.9529e-01, -1.0137e-01,  5.8423e-01,  6.7335e-01,
#builder.joint_target = [pos[0], pos[1], pos[2], quat[1], quat[2],
         #quat[3],  quat[0], -1.0137e-01,  5.8423e-01,  6.7335e-01,
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

sim_dt = 1e-5
sim_time = 0.0

###### OPTIMIZATION
base_joint_q = torch.clone(state.joint_q[inds])
self_contact_inds = torch.where(model.contact_body1 != 26)[0]

experiment_name = "cma"

def loss(base_joint_q):
    # SIMULATE WITH CONTACT FORCES
    state = model.state()
    state.joint_q[inds] = base_joint_q

    state.joint_qd[-1] = -0.01

    m = 1
    for k in range(m):
        state = integrator.forward(
            model, state, sim_dt,
            update_mass_matrix=True)

    l_task = 1e5*((state.joint_qd[-6:] - torch.zeros_like(state.joint_qd[-6:]))**2).sum()
    l_task = max(l_task,0.01)
    l_self_collision = 1e-7*(state.contact_f_s[self_contact_inds,:]**2).sum()
    l_joint = 1e-2*((base_joint_q[7:] - 0.5*(model.joint_limit_lower[inds][7:] + model.joint_limit_upper[inds][7:]))**2).sum()
    l = l_task + l_self_collision + l_joint + l_task
    return l

es = cma.CMAEvolutionStrategy(base_joint_q.detach().cpu().numpy(), 0.001, {'popsize': 100, 'conditioncov_alleviate': [1e14,1e14],
    'BoundaryHandler': cma.s.ch.BoundTransform, 'bounds': [[-np.inf]*7 + model.joint_limit_lower[7:].tolist(), [np.inf]*7 + model.joint_limit_upper[7:].tolist()]})

render = True
bests = []
i = 0
while not es.stop():
    solutions = es.ask()
    solutions_ = torch.tensor(solutions, dtype=torch.float, device=device)
    solutions_[:,3:7] /= torch.norm(solutions_[:,3:7],dim=1,keepdim=True)
    solutions_[:,7:] = torch.clamp(solutions_[:,7:], model.joint_limit_lower[7:base_joint_q.numel()][None,:], model.joint_limit_upper[7:base_joint_q.numel()][None,:])
    losses = [loss(solutions_[i,:]).item() for i in range(len(solutions))]
    es.tell(solutions, losses)
    es.logger.add()
    es.disp()
    es.timer.pause()
    if i % 1 == 0:
        bests.append(np.copy(es.result.xbest))
    i += 1

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

t = 0.0
for best in bests:
    state = model.state()
    base_joint_q = torch.tensor(best,dtype=torch.float,device=device)
    base_joint_q[3:7] /= torch.norm(base_joint_q[3:7])
    base_joint_q[7:] = torch.clamp(base_joint_q[7:], model.joint_limit_lower[7:base_joint_q.numel()], model.joint_limit_upper[7:base_joint_q.numel()])
    state.joint_q[inds] = base_joint_q

    m = 1
    for k in range(m):
        state = integrator.forward(
            model, state, sim_dt,
            update_mass_matrix=True)
    renderer.update(state, t)
    t += 1.0

if render:
    stage.Save()
    print(f"Saved USD stage at {stage_name}.")


# sample initial particles
n_particles = 1000
particles = torch.randn(n_particles,base_joint_q.numel()+contact_forces.numel(),device=device)
particles[:,len(inds):] *= 100.0
d_particles = particles.shape[1]
torch.random.manual_seed(0)
for i in range(n_particles):
    #particles[i,:base_joint_q.numel()] = base_joint_q
    quat = Quaternion.random()
    pos = -0.7*torch.tensor(quat.rotate([1.0,0.0,0.0]),device=device)
    particles[i,:3] = pos
    particles[i,3]=quat[1]
    particles[i,4]=quat[2]
    particles[i,5]=quat[3]
    particles[i,6]=quat[0]
particles[:,7:len(inds)] = torch.clamp(particles[:,7:len(inds)], model.joint_limit_lower[7:len(inds)][None,:], model.joint_limit_upper[7:len(inds)][None,:])
ratio = 1.0
particles /= ratio

n_components = 10
gmm=GaussianMixture(
    n_components=n_components,
    covariance_type="diag",
    tol=1e-32,
    reg_covar=1e-9,
    max_iter=10000,
    n_init=10,
    init_params="random",
    warm_start=False,
    verbose=0)
    #precisions_init=np.ones(particles.shape[1])[None,:]*0.01)
n_elite = 250
n_iters = 100000

render = True
if render:
    # set up Usd renderer
    from pxr import Usd
    from dflex.render import UsdRenderer

    renderers = []
    stages = []
    for i in range(n_components):
        stage_name = f"outputs/{experiment_name}_{i}.usd"
        stage = Usd.Stage.CreateNew(stage_name)
        renderer = UsdRenderer(model, stage)
        renderer.draw_points = False
        renderer.draw_springs = False
        renderer.draw_shapes = True
        renderers.append(renderer)
        stages.append(stage)

t = 0.0
old_mu = torch.zeros(n_components,particles.shape[1],device=device)
for i in range(n_iters):
    fitnesses = fitness(particles*ratio)
    _,elite_inds = torch.topk(fitnesses,n_elite)
    elite_particles = particles[elite_inds,:]
    print(f"iter {i} -- elite mean fitness = {fitnesses[elite_inds].max()}")
    gmm.fit(elite_particles.detach().cpu())
    gmm.covariances_ = gmm.covariances_.shape[0]*1* gmm.covariances_ / np.linalg.norm(gmm.covariances_)
    gmm.weights_ = gmm.weights_ / np.sum(gmm.weights_)
    #print(gmm.weights_)
    particles,y = gmm.sample(n_particles)
    particles = torch.tensor(particles,dtype=torch.float,device=device)
    particles[:,7:len(inds)] = torch.clamp(particles[:,7:len(inds)], model.joint_limit_lower[7:len(inds)][None,:], model.joint_limit_upper[7:len(inds)][None,:])
    particles[:,3:7] /= torch.norm(particles[:,3:7],dim=1)[:,None]

    if render and i % 1 == 0:
        mu = torch.tensor(gmm.means_,dtype=torch.float,device=device)
        mu[:,3:7] /= torch.norm(mu[:,3:7],dim=1)[:,None]
        print(torch.norm(old_mu[:,:len(inds)]-mu[:,:len(inds)]))
        for j in range(mu.shape[0]):
            base_joint_q = mu[j,:len(inds)]

            state = model.state()
            state.joint_q[inds] = base_joint_q

            m = 1
            for k in range(m):
                state = integrator.forward(
                    model, state, sim_dt,
                    update_mass_matrix=True)

            renderers[j].update(state, t)
        old_mu = mu
        t += 1.0

if render:
    for i in range(len(renderers)):
        stages[i].Save()
    print(f"Saved USD stage at {stage_name}.")


for j in range(n_particles):
    particle = particles[j,:]
    base_joint_q = particle[:len(inds)]

    state = model.state()
    state.joint_q[inds] = base_joint_q

    state.joint_qd[-2] = -0.01

    m = 1
    for k in range(m):
        state = integrator.forward(
            model, state, sim_dt,
            update_mass_matrix=True)

    renderer.update(state, t)
    t = t + 1.0

if render:
    stage.Save()
    print(f"Saved USD stage at {stage_name}.")


history = []
iteration = 0
for particles in CEM(fitness, particles, gmm_components=10,max_iter=30,alpha=0.01):
    history.append(particles)
    iteration += 1
    if iteration == 10:
        break

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

t = 0.0
for i in range(len(history)):
    for j in range(n_particles):
        particle = torch.tensor(history[i][j,:],dtype=torch.float,device=device)
        base_joint_q = particle[:len(inds)]

        state = model.state()
        state.joint_q[inds] = base_joint_q

        state.joint_qd[-1] = -0.01

        m = 1
        for k in range(m):
            state = integrator.forward(
                model, state, sim_dt,
                update_mass_matrix=True)

        renderer.update(state, t)
        t = t + 1.0

if render:
    stage.Save()
    print(f"Saved USD stage at {stage_name}.")


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

    state.joint_qd[-2] = -0.01

    m = 1
    for k in range(m):
        state = integrator.forward(
            model, state, sim_dt,
            update_mass_matrix=True)

    l_physics = 1e-3*((state.contact_f_s[box_contact_inds,:]+contact_forces)**2).sum()

    l_self_collision = 1e-7*(state.contact_f_s[self_contact_inds,:]**2).sum()

    l_joint = 1e-2*((base_joint_q[7:] - 0.5*(model.joint_limit_lower[inds][7:] + model.joint_limit_upper[inds][7:]))**2).sum()

    #forces = (state.contact_f_s[:]**2).sum(dim=1)
    #l_force = (torch.softmax(1e5*forces,dim=0)*forces).sum()

    l = l_physics + l_self_collision + l_joint

    mdmm_return = mdmm_module(l)

    if i % plot_every == 0:
        print(f"{mdmm_return.fn_values[0]} {l_self_collision} {l_joint} {l_physics} {base_joint_q} {contact_forces[0,:]}")

        if render:
            renderer.update(state, i / plot_every)
        if record_stats:
            #stats['c_balanced_forces'][i // plot_every] = mdmm_return.fn_values[0].detach().cpu()
            stats['c_drop'][i // plot_every] = mdmm_return.fn_values[0].detach().cpu()
            stats['l_force'][i // plot_every] = l_physics.detach().cpu()
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