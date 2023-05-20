import streamlit as st
from huggingface_hub import HfFolder
HfFolder().save_token(st.secrets['etoken'])


import numpy
import trimesh
import objaverse
import openshape
import misc_utils
import plotly.graph_objects as go


@st.cache_resource
def load_openshape(name):
    return openshape.load_pc_encoder(name)


f32 = numpy.float32
model_b32 = openshape.load_pc_encoder('openshape-pointbert-vitb32-rgb')
model_l14 = openshape.load_pc_encoder('openshape-pointbert-vitl14-rgb')
model_g14 = openshape.load_pc_encoder('openshape-pointbert-vitg14-rgb')


st.title("OpenShape Demo")
objaid = st.text_input("Enter an Objaverse ID")
model = st.file_uploader("Or upload a model (.glb/.obj/.ply)")
npy = st.file_uploader("Or upload a point cloud numpy array (.npy of Nx3 XYZ or Nx6 XYZRGB)")
swap_yz_axes = st.checkbox("Swap Y/Z axes of input (Y is up for OpenShape)")
prog = st.progress(0.0, "Idle")


def load_data():
    # load the model
    prog.progress(0.05, "Preparing Point Cloud")
    if npy is not None:
        pc: numpy.ndarray = numpy.load(npy)
    elif model is not None:
        pc = misc_utils.trimesh_to_pc(trimesh.load(model, model.name.split(".")[-1]))
    elif objaid:
        prog.progress(0.1, "Downloading Objaverse Object")
        objamodel = objaverse.load_objects([objaid])[objaid]
        prog.progress(0.2, "Preparing Point Cloud")
        pc = misc_utils.trimesh_to_pc(trimesh.load(objamodel))
    else:
        raise ValueError("You have to supply 3D input!")
    prog.progress(0.25, "Preprocessing Point Cloud")
    assert pc.ndim == 2, "invalid pc shape: ndim = %d != 2" % pc.ndim
    assert pc.shape[1] in [3, 6], "invalid pc shape: should have 3/6 channels, got %d" % pc.shape[1]
    if swap_yz_axes:
        pc[:, [1, 2]] = pc[:, [2, 1]]
    pc[:, :3] = pc[:, :3] - numpy.mean(pc[:, :3], axis=0)
    pc[:, :3] = pc[:, :3] / numpy.linalg.norm(pc[:, :3], axis=-1).max()
    if pc.shape[1] == 3:
        pc = numpy.concatenate([pc, numpy.ones_like(pc)], axis=-1)
    prog.progress(0.3, "Preprocessed Point Cloud")
    return pc.astype(f32)


def render_pc(pc):
    rand = numpy.random.permutation(len(pc))[:2048]
    pc = pc[rand]
    rgb = (pc[:, 3:] * 255).astype(numpy.uint8)
    g = go.Scatter3d(
        x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
        mode='markers',
        marker=dict(size=2, color=[f'rgb({rgb[i, 0]}, {rgb[i, 1]}, {rgb[i, 2]})' for i in range(len(pc))]),
    )
    fig = go.Figure(data=[g])
    fig.update_layout(scene_camera=dict(up=dict(x=0, y=1, z=0)))
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        # st.caption("Point Cloud Preview")
    return col2


try:
    if st.button("Run Classification on LVIS Categories"):
        pc = load_data()
        col2 = render_pc(pc)
        prog.progress(0.5, "Running Classification")
        pred = openshape.pred_lvis_sims(model_g14, pc)
        with col2:
            for i, (cat, sim) in zip(range(5), pred.items()):
                st.text(cat)
                st.caption("Similarity %.4f" % sim)
        prog.progress(1.0, "Idle")
except Exception as exc:
    st.error(repr(exc))
