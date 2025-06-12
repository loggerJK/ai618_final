# Installation
Use provided yaml file  local install the diffusers folder. Please let us know if you need further guide for environment setting.

```bash
conda env create -f guidance_ai618.yaml
conda activate guidance_ai618
cd ../diffusers
pip install -e ".[torch]"
```

# Run

### Generate images
Please refer to `sd3_inference.ipynb` file for example code.

```python
images = pipe(
    prompt,
    negative_prompt="",
    num_inference_steps=20,
    guidance_scale=0.0,
    generator=generator,
    return_dict=False,
    perturb_heads=perturb_heads,
    perturb_type=perturb_type,
    perturb_guidance_scale=perturb_guidance_scale,
)[0]
```

You can specify perturb_type, perturb_heads, perturb_guidance_scale. Refer to [Supported perturbations](#supported-perturbations) for more details.


# Supported perturbations
You can check implemented perturbations in `diffusers/src/diffusers/models/attention_processor.py` `def skip_attention` part.

### Attention map perturbations (after softmax)
- `[PROB_PERTURB]attention_identity`: replace the attention map with the identity matrix
- `[PROB_PERTURB]uniform`: replace the attention map with the uniform matrix
- `[PROB_PERTURB]II_identity_IT_mask`: replace the image-image attention map with the identity matrix and also make logit -inf of the image-text attention map

- `[PROB_PERTURB]zeroout`: zero out the image-image attention map
- `[PROB_PERTURB]II_identity`: replace the image-image attention map with the identity matrix
- `[PROB_PERTURB]II_mask`
- `[PROB_PERTURB]IT_mask`
- `[PROB_PERTURB]TI_mask`
- `[PROB_PERTURB]TT_mask`
- `[PROB_PERTURB]TT_identity`
- `[PROB_PERTURB]II_identity_renormalize`

### Attention logit perturbations (before softmax)
- `[LOGIT_PERTURB]II_identity`
- `[LOGIT_PERTURB]II_mask`
- `[LOGIT_PERTURB]IT_mask`
- `[LOGIT_PERTURB]TI_mask`
- `[LOGIT_PERTURB]TT_mask`
- `[LOGIT_PERTURB]TT_identity`
- `smoothed_energy@all`
- `smoothed_energy@img`
- `smoothed_energy@txt`


You can also specify the identity scale for the perturbations.
`"[PROB_PERTURB]II_identity@scale={identity_scale}"`

