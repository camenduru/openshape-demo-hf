import streamlit as st
from huggingface_hub import HfFolder, snapshot_download
HfFolder().save_token(st.secrets['etoken'])
snapshot_download("OpenShape/openshape-demo-support", local_dir='.')


import numpy
import openshape
from openshape.demo import misc_utils, classification


@st.cache_resource
def load_openshape(name):
    return openshape.load_pc_encoder(name)


f32 = numpy.float32
# clip_model, clip_prep = load_openclip()
model_g14 = openshape.load_pc_encoder('openshape-pointbert-vitg14-rgb')


st.title("OpenShape Demo")
load_data = misc_utils.input_3d_shape()
prog = st.progress(0.0, "Idle")


try:
    if st.button("Run Classification on LVIS Categories"):
        pc = load_data(prog)
        col2 = misc_utils.render_pc(pc)
        prog.progress(0.5, "Running Classification")
        pred = classification.pred_lvis_sims(model_g14, pc)
        with col2:
            for i, (cat, sim) in zip(range(5), pred.items()):
                st.text(cat)
                st.caption("Similarity %.4f" % sim)
        prog.progress(1.0, "Idle")
except Exception as exc:
    st.error(repr(exc))
