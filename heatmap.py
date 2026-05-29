import numpy as np
import plotly.graph_objects as go

file_path = "data/train/sagittal/0000.npy"
volume = np.load(file_path).astype(np.float32)

print("Original shape:", volume.shape)

# Downsample for browser speed
vol = volume[:, ::4, ::4]

# Normalize
vol = (vol - vol.min()) / (vol.max() - vol.min())

z, y, x = np.mgrid[
    0:vol.shape[0],
    0:vol.shape[1],
    0:vol.shape[2]
]

# Fake "important region" near center slices
important_start = vol.shape[0] // 2 - 2
important_end = vol.shape[0] // 2 + 3

heatmap = np.zeros_like(vol)
heatmap[important_start:important_end, 20:45, 20:45] = 1.0

fig = go.Figure()

# MRI volume
fig.add_trace(go.Volume(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=vol.flatten(),
    opacity=0.10,
    surface_count=15,
    colorscale="Gray",
    name="MRI Volume"
))

# Heatmap / highlighted region
fig.add_trace(go.Volume(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=heatmap.flatten(),
    opacity=0.35,
    surface_count=3,
    colorscale="Hot",
    showscale=False,
    name="Important ACL Region"
))

fig.update_layout(
    title="3D Knee MRI with Simulated ACL Attention Heatmap",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Slice",
        aspectmode="data"
    )
)

fig.write_html("outputs/3d_acl_heatmap_demo.html", auto_open=True)