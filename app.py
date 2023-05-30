import sys
import streamlit as st
from huggingface_hub import HfFolder, snapshot_download


@st.cache_data
def load_support():
    if st.secrets.has_key('etoken'):
        HfFolder().save_token(st.secrets['etoken'])
    sys.path.append(snapshot_download("OpenShape/openshape-demo-support"))


# st.set_page_config(layout='wide')
load_support()


import numpy
import torch
import openshape
import transformers
from PIL import Image

@st.cache_resource
def load_openshape(name):
    return openshape.load_pc_encoder(name)


@st.cache_resource
def load_openclip():
    return transformers.CLIPModel.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        low_cpu_mem_usage=True, torch_dtype=half,
        offload_state_dict=True
    ), transformers.CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")


f32 = numpy.float32
half = torch.float16 if torch.cuda.is_available() else torch.bfloat16
# clip_model, clip_prep = None, None
clip_model, clip_prep = load_openclip()
model_b32 = load_openshape('openshape-pointbert-vitb32-rgb').cpu()
model_l14 = load_openshape('openshape-pointbert-vitl14-rgb')
model_g14 = load_openshape('openshape-pointbert-vitg14-rgb')
torch.set_grad_enabled(False)

from openshape.demo import misc_utils, classification, caption, sd_pc2img, retrieval


st.title("OpenShape Demo")
st.caption("For faster inference without waiting in queue, you may clone the space and run it yourself.")
prog = st.progress(0.0, "Idle")
tab_cls, tab_img, tab_text, tab_pc, tab_sd, tab_cap = st.tabs([
    "Classification",
    "Retrieval w/ Image",
    "Retrieval w/ Text",
    "Retrieval w/ 3D",
    "Image Generation",
    "Captioning",
])


def demo_classification():
    load_data = misc_utils.input_3d_shape('cls')
    cats = st.text_input("Custom Categories (64 max, separated with comma)")
    cats = [a.strip() for a in cats.split(',')]
    if len(cats) > 64:
        st.error('Maximum 64 custom categories supported in the demo')
        return
    lvis_run = st.button("Run Classification on LVIS Categories")
    custom_run = st.button("Run Classification on Custom Categories")
    if lvis_run:
        pc = load_data(prog)
        col2 = misc_utils.render_pc(pc)
        prog.progress(0.5, "Running Classification")
        pred = classification.pred_lvis_sims(model_g14, pc)
        with col2:
            for i, (cat, sim) in zip(range(5), pred.items()):
                st.text(cat)
                st.caption("Similarity %.4f" % sim)
        prog.progress(1.0, "Idle")
    if custom_run:
        pc = load_data(prog)
        col2 = misc_utils.render_pc(pc)
        prog.progress(0.5, "Computing Category Embeddings")
        device = clip_model.device
        tn = clip_prep(text=cats, return_tensors='pt', truncation=True, max_length=76).to(device)
        feats = clip_model.get_text_features(**tn).float().cpu()
        prog.progress(0.5, "Running Classification")
        pred = classification.pred_custom_sims(model_g14, pc, cats, feats)
        with col2:
            for i, (cat, sim) in zip(range(5), pred.items()):
                st.text(cat)
                st.caption("Similarity %.4f" % sim)
        prog.progress(1.0, "Idle")


def demo_captioning():
    with st.form("capform"):
        load_data = misc_utils.input_3d_shape('cap')
        cond_scale = st.slider('Conditioning Scale', 0.0, 4.0, 2.0)
        if st.form_submit_button("Generate a Caption"):
            pc = load_data(prog)
            col2 = misc_utils.render_pc(pc)
            prog.progress(0.5, "Running Generation")
            cap = caption.pc_caption(model_b32, pc, cond_scale)
            st.text(cap)
            prog.progress(1.0, "Idle")


def demo_pc2img():
    with st.form("sdform"):
        load_data = misc_utils.input_3d_shape('sd')
        prompt = st.text_input("Prompt (Optional)")
        noise_scale = st.slider('Variation Level', 0, 5, 1)
        cfg_scale = st.slider('Guidance Scale', 0.0, 30.0, 10.0)
        steps = st.slider('Diffusion Steps', 8, 50, 25)
        width = 640  # st.slider('Width', 480, 640, step=32)
        height = 640  # st.slider('Height', 480, 640, step=32)
        if st.form_submit_button("Generate"):
            pc = load_data(prog)
            col2 = misc_utils.render_pc(pc)
            prog.progress(0.49, "Running Generation")
            if torch.cuda.is_available():
                clip_model.cpu()
            img = sd_pc2img.pc_to_image(
                model_l14, pc, prompt, noise_scale, width, height, cfg_scale, steps,
                lambda i, t, _: prog.progress(0.49 + i / (steps + 1) / 2, "Running Diffusion Step %d" % i)
            )
            if torch.cuda.is_available():
                clip_model.cuda()
            with col2:
                st.image(img)
            prog.progress(1.0, "Idle")


def retrieval_results(results):
    for i in range(len(results) // 4):
        cols = st.columns(4)
        for j in range(4):
            idx = i * 4 + j
            if idx >= len(results):
                continue
            entry = results[idx]
            with cols[j]:
                ext_link = f"https://objaverse.allenai.org/explore/?query={entry['u']}"
                st.image(entry['img'])
                # st.markdown(f"[![thumbnail {entry['desc'].replace('\n', ' ')}]({entry['img']})]({ext_link})")
                # st.text(entry['name'])
                quote_name = entry['name'].replace('[', '\\[').replace(']', '\\]').replace('\n', ' ')
                st.markdown(f"[{quote_name}]({ext_link})")


def demo_retrieval():
    with tab_text:
        with st.form("rtextform"):
            k = st.slider("# Shapes to Retrieve", 1, 100, 16, key='rtext')
            text = st.text_input("Input Text")
            if st.form_submit_button("Run with Text"):
                prog.progress(0.49, "Computing Embeddings")
                device = clip_model.device
                tn = clip_prep(text=[text], return_tensors='pt', truncation=True, max_length=76).to(device)
                enc = clip_model.get_text_features(**tn).float().cpu()
                prog.progress(0.7, "Running Retrieval")
                retrieval_results(retrieval.retrieve(enc, k))
                prog.progress(1.0, "Idle")

    with tab_img:
        with st.form("rimgform"):
            k = st.slider("# Shapes to Retrieve", 1, 100, 16, key='rimage')
            pic = st.file_uploader("Upload an Image")
            if st.form_submit_button("Run with Image"):
                img = Image.open(pic)
                st.image(img)
                prog.progress(0.49, "Computing Embeddings")
                device = clip_model.device
                tn = clip_prep(images=[img], return_tensors="pt").to(device)
                enc = clip_model.get_image_features(pixel_values=tn['pixel_values'].type(half)).float().cpu()
                prog.progress(0.7, "Running Retrieval")
                retrieval_results(retrieval.retrieve(enc, k))
                prog.progress(1.0, "Idle")

    with tab_pc:
        with st.form("rpcform"):
            k = st.slider("# Shapes to Retrieve", 1, 100, 16, key='rpc')
            load_data = misc_utils.input_3d_shape('retpc')
            if st.form_submit_button("Run with Shape"):
                pc = load_data(prog)
                col2 = misc_utils.render_pc(pc)
                prog.progress(0.49, "Computing Embeddings")
                ref_dev = next(model_g14.parameters()).device
                enc = model_g14(torch.tensor(pc[:, [0, 2, 1, 3, 4, 5]].T[None], device=ref_dev)).cpu()
                prog.progress(0.7, "Running Retrieval")
                retrieval_results(retrieval.retrieve(enc, k))
                prog.progress(1.0, "Idle")


try:
    if torch.cuda.is_available():
        clip_model.cuda()
    with tab_cls:
        demo_classification()
    with tab_cap:
        demo_captioning()
    with tab_sd:
        demo_pc2img()
    demo_retrieval()
except Exception:
    import traceback
    st.error(traceback.format_exc().replace("\n", "  \n"))
