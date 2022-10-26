import hydra
from matplotlib.pyplot import get
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from hydra.utils import to_absolute_path
from manotorch.manolayer import ManoLayer

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src'))

from cio import CIO
from grad import GradOpt

@hydra.main(config_path="../conf/collect_grasps", config_name="config")
def collect_grasps(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # here we loop through objects and random starts
    if cfg.collector_config.type in ["cio", "grad"]:
        if cfg.collector_config.type == "cio":
            collector = CIO(cfg.collector_config)
        else:
            collector = GradOpt(cfg.collector_config)

        results = {}

        obj_config = cfg.collector_config.object
        for i in range(cfg.collector_config.n_starts):
            collector.build_model(obj_config, with_viewer_mesh=True)
            obj_name = obj_config.name.replace("/","_")
            exp_name = f"{obj_name}_{i}_coarse_to_fine_{cfg.collector_config.coarse_to_fine}_{cfg.collector_config.type}_scale_{obj_config.rescale}"
            initial_guess = collector.sample_initial_guess()
            result = collector.run(initial_guess)
            results[exp_name] = result

            
            collector.build_model(obj_config, with_viewer_mesh=True)
            # mano_layer = ManoLayer(
            #     mano_assets_root=to_absolute_path('grasping/data/assets_mano'),
            #     use_pca=False).to(collector.device)

            if cfg.render_final:
                # set up Usd renderer
                from pxr import Usd
                from dflex.render import UsdRenderer
                stage_name = f"outputs/{exp_name}_final.usd"
                stage = Usd.Stage.CreateNew(stage_name)
                renderer = UsdRenderer(collector.model, stage)
                renderer.draw_points = False
                renderer.draw_springs = False
                renderer.draw_shapes = True

                state = collector.model.state()
                state.joint_q[:] = result['final_joint_q']
                state = collector.integrator.forward(
                    collector.model, state, 1e-5,
                    update_mass_matrix=True)

                mano_q = result['final_mano_q']
                mano_q = mano_q.to(collector.device)
                #mano_joints = result['history']['joint_angles'][j,:,:].to(collector.device)
                mano_output = collector.mano_layer(mano_q[None,:], collector.mano_shape)
                #mano_output = mano_layer(mano_joints.flatten()[None,:48],collector.mano_shape)
                vertices = mano_output.verts.cpu()
                vertices *= collector.mano_ratio
                m=stage.GetObjectAtPath("/root/body_0/mesh_0")
                m.GetAttribute("points").Set(vertices[0,:,:].detach().cpu().numpy(),0.0)

                renderer.update(state, 0.0)
                stage.Save()
                print(f"Saved USD stage at {stage_name}.")

            if cfg.render_initial:
                # set up Usd renderer
                from pxr import Usd
                from dflex.render import UsdRenderer
                stage_name = f"outputs/{exp_name}_initial.usd"
                stage = Usd.Stage.CreateNew(stage_name)
                renderer = UsdRenderer(collector.model, stage)
                renderer.draw_points = False
                renderer.draw_springs = False
                renderer.draw_shapes = True

                state = collector.model.state()
                state.joint_q[:] = result['history']['joint_q'][0,:]
                state = collector.integrator.forward(
                    collector.model, state, 1e-5,
                    update_mass_matrix=True)

                mano_q = result['history']['mano_q'][0,:]
                mano_q = mano_q.to(collector.device)
                #mano_joints = result['history']['joint_angles'][j,:,:].to(collector.device)
                mano_output = collector.mano_layer(mano_q[None,:], collector.mano_shape)
                #mano_output = mano_layer(mano_joints.flatten()[None,:48],collector.mano_shape)
                vertices = mano_output.verts.cpu()
                vertices *= collector.mano_ratio
                m=stage.GetObjectAtPath("/root/body_0/mesh_0")
                m.GetAttribute("points").Set(vertices[0,:,:].detach().cpu().numpy(),0.0)

                renderer.update(state, 0.0)
                stage.Save()
                print(f"Saved USD stage at {stage_name}.")

            if cfg.render_all:
                # set up Usd renderer
                from pxr import Usd
                from dflex.render import UsdRenderer
                stage_name = f"outputs/{exp_name}_all.usd"
                stage = Usd.Stage.CreateNew(stage_name)
                renderer = UsdRenderer(collector.model, stage)
                renderer.draw_points = False
                renderer.draw_springs = False
                renderer.draw_shapes = True

                state = collector.model.state()
                
                sim_t = 0.0
                for j in range(result['history']['joint_q'].shape[0]):
                    state.joint_q[:] = result['history']['joint_q'][j,:]
                    #state.joint_q[:3] =  torch.tensor([-0.0947480668596128, -0.10516723154368249, 0.044520795511424655],device=collector.device)
                    #grasp = np.load(to_absolute_path("grasping/data/grasps/obman/02876657_1a7ba1f4c892e2da30711cdbdbc73924_scale_125.0_0.npy"),allow_pickle=True).item()
                    #state.joint_q[:] = torch.tensor(grasp["final_joint_q"],dtype=torch.float32,device=collector.device)
                    state = collector.integrator.forward(
                        collector.model, state, 1e-5,
                        update_mass_matrix=True)

                    mano_q = result['history']['mano_q'][j,:]
                    #mano_q = torch.tensor(grasp["final_mano_q"],dtype=torch.float32,device=collector.device)
                    mano_q = mano_q.to(collector.device)
                    #mano_q = [-1.573274162744035, 0.18210576238566262, -2.401482457417205, 0.2676777430581665, 0.11828904112982677, 0.921057652386048, 0.20569340778159206, -0.0016220880874205175, 0.9202578415766811, 0.12341604466895524, -0.0009732528524523103, 0.5521547049460087, 0.0, 0.0, 0.8242170748005676, 0.0, 0.0, 0.8242170748005675, 0.0, 0.0, 0.4945302448803405, -0.2852258412466123, 0.02257015354873201, 0.45940785030794234, -0.5056287291845366, -0.01018721759470846, 0.8253460284746491, -0.3033772375107219, -0.006112330556825091, 0.49520761708478944, -0.17693655527723623, -0.2023554710847482, 0.7102934557617074, -0.10145149608481938, -0.0004987883421543483, 0.7334840380104062, -0.06087089765089164, -0.0002992730052926336, 0.4400904228062438, 0.9950741343857169, 0.13169240503739693, 0.7358209192038518, -0.02246915642154966, -0.33119153811544355, 0.018500291812672636, 0.07119197549458654, -0.28519550595126214, 0.15531541698841728]
                    #mano_q = torch.tensor(mano_q,device=collector.device)
                    #mano_output = mano_layer(mano_q[None,:],collector.mano_shape)
                    mano_output = collector.mano_layer(mano_q[None,:], collector.mano_shape)
                    vertices = mano_output.verts.cpu()
                    vertices *= collector.mano_ratio
                    m=stage.GetObjectAtPath("/root/body_0/mesh_0")
                    m.GetAttribute("points").Set(vertices[0,:,:].detach().cpu().numpy(),sim_t)

                    renderer.update(state, sim_t)
                    sim_t += 1.0
                stage.Save()
                print(f"Saved USD stage at {stage_name}.")

            result_filename = f"outputs/{exp_name}.npy"
            np.save(result_filename, dict(result=result, obj_config=obj_config, name=exp_name))
            print(f"Saved results npy at {result_filename}.")

if __name__ == "__main__":
    collect_grasps()
