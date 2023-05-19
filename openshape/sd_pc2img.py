import numpy
import torch
import torch_redstone as rst
import transformers
from diffusers import StableUnCLIPImg2ImgPipeline


class Wrapper(transformers.modeling_utils.PreTrainedModel):
    def __init__(self) -> None:
        super().__init__(transformers.configuration_utils.PretrainedConfig())
        self.param = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return rst.ObjectProxy(image_embeds=x)


pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "diffusers/stable-diffusion-2-1-unclip-i2i-l",
    image_encoder = Wrapper()
)
if torch.cuda.is_available():
    pipe = pipe.to('cuda:' + str(torch.cuda.current_device()))
    pipe.enable_model_cpu_offload(torch.cuda.current_device())


@torch.no_grad()
def pc_to_image(pc_encoder: torch.nn.Module, pc, prompt, noise_level, width, height, cfg_scale, num_steps, callback):
    ref_dev = next(pc_encoder.parameters()).device
    enc = pc_encoder(torch.tensor(pc.T[None], device=ref_dev))
    return pipe(
        prompt="best quality, super high resolution, " + prompt,
        negative_prompt="cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        image=torch.nn.functional.normalize(enc, dim=-1) * (768 ** 0.5) / 2,
        width=width, height=height,
        guidance_scale=cfg_scale,
        noise_level=noise_level,
        callback=callback,
        num_inference_steps=num_steps
    ).images[0]
