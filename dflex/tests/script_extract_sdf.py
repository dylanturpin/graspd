import yaml
import numpy as np
import torch

#sdf_path = "data/025_mug_16k_256_scale.yaml"
#sdf_path = "data/sphere/sphere_sdf.yaml"
#sdf_path = "data/stretched_sphere/stretched_sphere_sdf.yaml"
#sdf_path = "data/025_mug/025_mug_16k_128.yaml"
#sdf_path = "data/025_mug/025_mug_16k_128_origscale_025pad_sdf.yaml"
#sdf_path = "data/025_mug/025_mug_16k_128_scale10_01pad_sdf.yaml"
sdf_path = "data/025_mug/025_mug_16k_256_scale10_01pad_sdf.yaml"
out_path = sdf_path[:-4] + "npy"
with open(sdf_path) as f:
    data = f.readlines()    

    sdf_start = data.index("  m_samples:\n")+1
    sdf_end = data.index("  m_packedUVs:\n")
    n = sdf_end - sdf_start

    sdf = np.zeros(n)
    for i in range(n):
        sdf[i] = float(data[sdf_start+i][3:])
        if i % 1000 == 0:
            print(i)

    # last 5 lines with admin data... data[-5:]
    last_five = [l[2:] for l in data[-5:]]
    last_five = "".join(last_five)
    metadata = yaml.load(last_five, Loader=yaml.CLoader)
    min_bounds = np.array([metadata["m_minBounds"]["x"], metadata["m_minBounds"]["y"], metadata["m_minBounds"]["z"]])
    max_bounds = np.array([metadata["m_maxBounds"]["x"], metadata["m_maxBounds"]["y"], metadata["m_maxBounds"]["z"]])

    # flip x 
    tmp = min_bounds[0]
    min_bounds[0] = -max_bounds[0]
    max_bounds[0] = -tmp
    sdf = np.flip(sdf.reshape((256,256,256),order="F"),axis=0).flatten(order="F")

    pos = (min_bounds + max_bounds)/2
    scale = max_bounds - min_bounds

    np.save(out_path, dict(sdf=sdf, pos=pos, scale=scale), allow_pickle=True)
