import nibabel as nb
from neuromaps.datasets import fetch_fslr
from surfplot import Plot
import numpy as np
import matplotlib.pyplot as plt
import sys;sys.path.append("../../")
import utils
import HMA
import pandas as pd


# Load GIFTI label file (left hemisphere)
lh_labels_gii = nb.load('../AAL.32k.L.label.gii')
lh_labels = lh_labels_gii.darrays[0].data.astype(int)

# Load GIFTI label file (right hemisphere)
rh_labels_gii = nb.load('../AAL.32k.R.label.gii')
rh_labels = rh_labels_gii.darrays[0].data.astype(int)

#%%

fig_width_px = 4500
fig_height_px = 2400
#%%
# struct = np.loadtxt("../structural_Deco_AAL.txt")
maps = np.load("maps2brain.npz")
mapnames = maps.files
AALlabels = pd.read_csv("../../../sorted_AAL_labels.txt")["label"].values #nombre de las areas
print(AALlabels[np.argsort(maps["Hin_node"])])

#%%
# mapnames = ("Hin_node","klsCNT_broken","klsMCS_broken","klsUWS_broken")
for n,name in enumerate(mapnames):
    mapp = maps[name]
    vector_map = utils.reord(mapp,do="LlrR to LlRr")+1e-6
    # Map data to vertices
    lh_vertex_data = np.zeros_like(lh_labels, dtype=float)
    rh_vertex_data = np.zeros_like(rh_labels, dtype=float)
    
    for i in range(45):
        lh_vertex_data[lh_labels == i + 1] = vector_map[i]
        rh_vertex_data[rh_labels == i + 1] = vector_map[i + 45]
    
    # Normalize data (optional, but recommended for consistent color scaling)
    maxi = np.max(np.abs(np.append(lh_vertex_data, rh_vertex_data)))
    if maxi > 0: # Avoid division by zero if all values are zero
        lh_vertex_data /= maxi
        rh_vertex_data /= maxi
    

    # Plotting
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']
    p = Plot(lh, rh, views=["lateral","medial"], zoom=1.5, size = (1000,800))
    p.add_layer({'left': lh_vertex_data, 'right': rh_vertex_data}, cmap='hot_r', cbar=True, color_range=(0, 1))
    fig = p.build()
    fig.savefig(f"preplot_{name}.svg", dpi=300, transparent=True)
    # fig.savefig(f"preplot_{name}.svg")
# plt.close(fig)