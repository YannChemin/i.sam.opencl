## DESCRIPTION

*i.sam.opencl* segments a 3-band imagery group with the Segment Anything
Model (SAM) and writes the segments as a clean vector area map. The whole
network, image encoder and mask decoder, runs on an OpenCL device (GPU or
CPU), with weights read directly from the original PyTorch checkpoint
file: neither Python nor PyTorch is needed at run time.

The module implements SAM's automatic mask generation: the model is
prompted with a regular grid of points, and the masks it returns are
kept according to their predicted quality and stability, de-duplicated
and cleaned. The overlapping masks are then turned into a
non-overlapping, topologically clean vector map with one attribute row
per segment.

The **group** (or its **subgroup**) must contain exactly three raster
maps, which are used as the red, green and blue channels in the order
they are listed in the group (see *i.group*). Each band is stretched
linearly between its lower and upper **percentile** to the 0-255 range
SAM was trained on. Cells that are NULL in any band are not segmented.
Processing is done in the current computational region.

### Checkpoint

The **checkpoint** is one of the original model files published with
SAM; the model variant is detected from the tensor shapes:

- ViT-H: [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
  (2.4 GiB, best quality),
- ViT-L: [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
  (1.2 GiB),
- ViT-B: [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
  (358 MiB, fastest).

The weights are not installed with the module. Instead of one
**checkpoint** file, **weights_dir** can name the folder where the
checkpoints were downloaded. A checkpoint is recognised there by
`vit_h`, `vit_l` or `vit_b` in its `.pth` file name, so renamed copies
such as `sam_vit_h_checkpoint.pth` are found too. The best model present
is used (ViT-H, then ViT-L, then ViT-B), or the one chosen with
**model**; the module checks that the file really holds that model. If a
folder holds several files of the chosen model, the first in
alphabetical order is used, with a warning.

### Mask generation

The region is processed in square crops. A region no larger than
**tile_size** cells is one crop, a larger region is split into tiles of
**tile_size** cells overlapping by **tile_overlap** cells. With
**crop_layers** greater than 0, each tile is additionally segmented in
zoomed-in sub-crops (4 in layer 1, 16 in layer 2, ...), prompted with
**points_per_side** divided by **crop_points_downscale** per layer.

Each crop is resized so that its longest side is 1024 pixels and passed
through the image encoder. The mask decoder is then prompted with a grid
of **points_per_side** x **points_per_side** points, **batch** prompts at
a time, returning three candidate masks per point. A mask is kept if its
predicted IoU exceeds **iou_threshold** and its stability score (the IoU
of the mask thresholded at plus and minus **stability_offset**) is at
least **stability_threshold**. Duplicates whose bounding boxes overlap
by more than **nms_threshold** (IoU) are suppressed, keeping the best
one. Masks touching an inner tile or crop edge are discarded, since the
neighbouring tile sees them whole. Holes and islands smaller than
**minsize** cells are finally removed from each mask.

These steps and their defaults are those of the reference
`SamAutomaticMaskGenerator`; on identical input, *i.sam.opencl* returns
the same masks.

### Vectorization and cleaning

SAM masks overlap: a field and the parcels inside it can both be masks.
The **overlap** rule chooses which mask a cell belongs to: the smallest
covering mask (*small*, finest partition, the default), the largest
(*large*) or the one with the highest predicted IoU (*score*). Masks
larger than **maxsize** cells are discarded first; this is the way to
drop the single mask SAM often returns for the whole background of a
scene.

The resulting label raster is cleaned: every connected piece smaller
than **minsize** cells, including small unsegmented gaps, is merged into
the neighbouring piece it shares the longest border with. Boundaries
are traced along cell edges into shared topological boundaries,
simplified with the Douglas-Peucker algorithm using the **simplify**
tolerance (one cell by default; 0 keeps the raster staircase), and the
topology is cleaned (boundaries broken at intersections, duplicates
removed, pseudo-nodes merged, slivers smaller than one cell removed).
Each segment gets one centroid, placed at the cell farthest from its
border.

Areas without a centroid are gaps that no mask covers. The optional
**raster** output holds the segment categories of the cleaned label
raster before simplification.

### Attribute table

| Column        | Description                                        |
| ------------- | -------------------------------------------------- |
| cat           | Segment category                                   |
| mask          | Number of the SAM mask the segment comes from      |
| predicted_iou | Mask quality predicted by SAM                      |
| stability     | Stability score of the mask                        |
| cells         | Segment size in cells                              |
| area          | Area of the simplified polygon in square map units |

A mask split into several pieces by overlapping masks yields several
segments with the same **mask** number.

## NOTES

### OpenCL device

With **device**=*auto*, the first GPU is used, else the first CPU
device. When several OpenCL platforms expose the same device (for
example Mesa's Clover and rusticl drivers for AMD GPUs), the first
platform on which the kernels compile is used; **platform** restricts
the choice to platforms whose name contains the given text. Mesa's
rusticl only exposes a GPU when enabled in the environment, e.g.
`RUSTICL_ENABLE=radeonsi`. A CPU OpenCL implementation such as PoCL
gives the same result but is about two orders of magnitude slower.

The ViT-H weights take 2.4 GiB of device memory and the work buffers
about 1.3 GiB with the default **batch** of 32 prompts; a smaller
**batch** reduces memory use. Buffers are kept below the device's
maximum allocation size.

### Performance

On an AMD Radeon Pro WX 7100 (Polaris, 16 GiB, Mesa rusticl), with the
ViT-H model, loading the model takes 1-2 s, the image encoder 3.8 s per
crop and the mask decoder 3.7 s per 1024 prompts (the default 32 x 32
grid). A single-crop region is thus segmented in about 8 s. The number
of crops, and so the run time, grows with the region size divided by
**tile_size**, and with **crop_layers**.

### Choosing the resolution

SAM sees every crop at 1024 x 1024 pixels. Objects should span at least
ten or so pixels at that scale: set the region resolution and
**tile_size** accordingly. A smaller **tile_size** zooms in (crops are
upsampled to 1024 pixels), which finds smaller objects at the cost of
more crops. For agricultural parcels in 2.5 to 10 m imagery, the
defaults work well.

### Debugging aids

When the environment variable `I_SAM_OPENCL_DUMP` names a directory,
the 8-bit crops, the encoder inputs, the image embeddings and the list
of masks are written there. The script `tests/reference_compare.py` in
the source tree compares them with the PyTorch reference
implementation. `I_SAM_OPENCL_PROFILE=1` prints the time spent in each
OpenCL kernel.

## EXAMPLES

### Landsat scene (North Carolina sample dataset)

Segment a true-colour composite of Landsat 7 bands 3, 2 and 1, dropping
the background mask (the region has 475 x 527 cells):

```sh
g.region raster=lsat7_2002_30 -p
i.group group=lsat_rgb input=lsat7_2002_30,lsat7_2002_20,lsat7_2002_10
i.sam.opencl group=lsat_rgb checkpoint=sam_vit_h_4b8939.pth \
    output=lsat_segments maxsize=50000
v.db.select lsat_segments | head
```

![i.sam.opencl example](i.sam.opencl.png)  
*Segments of the Landsat 7 true-colour composite (yellow).*

### Weights folder

With the three checkpoints downloaded into `~/models/sam`, use the
fastest model for a quick look:

```sh
i.sam.opencl group=lsat_rgb weights_dir=~/models/sam model=vit_b \
    output=lsat_quick maxsize=50000
```

### Agricultural parcels

Delineate fields in a 2.5 m VNIR image with bands 1-3 grouped as RGB,
keeping only segments of at least 0.1 ha (160 cells) and a label raster
alongside the vector map:

```sh
i.group group=vnir_rgb input=vnir.1,vnir.2,vnir.3
g.region raster=vnir.1 -p
i.sam.opencl group=vnir_rgb checkpoint=sam_vit_h_4b8939.pth \
    output=fields raster=fields minsize=160
```

## REFERENCES

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L.,
Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollar, P., Girshick, R.,
2023. Segment Anything. Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), 4015-4026.
[arXiv:2304.02643](https://arxiv.org/abs/2304.02643)

## SEE ALSO

*[i.group](i.group.md), [i.segment](i.segment.md),
[r.to.vect](r.to.vect.md), [v.clean](v.clean.md),
[v.generalize](v.generalize.md)*

## AUTHORS

Yann Chemin
