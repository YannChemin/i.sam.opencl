# i.sam.opencl

GRASS GIS module that segments a 3-band imagery group with Meta's Segment
Anything Model (SAM) on an OpenCL device (GPU or CPU) and writes the
segments as a clean vector area map. The whole network runs in OpenCL and
reads the original PyTorch checkpoint directly: neither Python nor PyTorch
is needed at run time. See `i.sam.opencl.md` for the full manual.

## SAM weights

The weights are not shipped with the module (they are far too large for a
source repository). Download at least one of the three original
checkpoints published by Meta with Segment Anything (Apache 2.0 licence):

| Model | File | Size | Download |
| ----- | ---- | ---- | -------- |
| ViT-H (best quality) | `sam_vit_h_4b8939.pth` | 2.4 GiB (2 564 550 879 bytes) | <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth> |
| ViT-L | `sam_vit_l_0b3195.pth` | 1.2 GiB (1 249 524 607 bytes) | <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth> |
| ViT-B (fastest) | `sam_vit_b_01ec64.pth` | 358 MiB (375 042 383 bytes) | <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth> |

For example, into `~/models/sam`:

```sh
mkdir -p ~/models/sam
cd ~/models/sam
for f in sam_vit_h_4b8939 sam_vit_l_0b3195 sam_vit_b_01ec64; do
    curl -L -O https://dl.fbaipublicfiles.com/segment_anything/$f.pth
done
```

The checkpoint is given to the module either as a file or as the folder
holding the downloaded files:

```sh
# One checkpoint file:
i.sam.opencl group=rgb checkpoint=~/models/sam/sam_vit_h_4b8939.pth output=segments

# The download folder: the best model found there is used (ViT-H, then
# ViT-L, then ViT-B), or the one chosen with model=
i.sam.opencl group=rgb weights_dir=~/models/sam output=segments
i.sam.opencl group=rgb weights_dir=~/models/sam model=vit_b output=segments
```

In a folder, a checkpoint is recognised by `vit_h`, `vit_l` or `vit_b`
in its `.pth` file name, so renamed copies such as
`sam_vit_h_checkpoint.pth` are found too. The model variant is always
checked against the tensor shapes inside the file.

## Building

```sh
make MODULE_TOPDIR=$HOME/dev/grass
```

or install into the user's addon directory with
`g.extension extension=i.sam.opencl url=/path/to/i.sam.opencl`.
Requires an OpenCL 1.2 platform and its development headers (e.g.
`ocl-icd-opencl-dev`) and a GRASS GIS build with OpenCL support. With Mesa
on AMD GPUs, export `RUSTICL_ENABLE=radeonsi` (or the matching driver) to
expose the GPU through rusticl.

## Tests

```sh
I_SAM_OPENCL_CHECKPOINT=~/models/sam/sam_vit_h_4b8939.pth pytest tests
```

Without `I_SAM_OPENCL_CHECKPOINT`, only the input validation tests run.
