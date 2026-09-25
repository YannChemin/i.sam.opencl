#ifndef I_SAM_OPENCL_IMAGE_H
#define I_SAM_OPENCL_IMAGE_H

#include <grass/gis.h>

/* The current region as an 8-bit RGB image. */
struct sam_image {
    int rows, cols;
    unsigned char *rgb;   /* rows * cols * 3, band order of the group. */
    unsigned char *valid; /* rows * cols, 0 where any band is NULL. */
    char names[3][GNAME_MAX + GMAPSET_MAX + 1];
};

/* Read the three raster maps of an imagery group (or of one of its
 * subgroups when subgroup is not NULL) over the current region and
 * stretch each band linearly between its lo and hi percentiles to
 * 0..255. Fatal unless the (sub)group holds exactly three maps. */
void image_read_group(const char *group, const char *subgroup, double lo,
                      double hi, struct sam_image *img);

/* Build the SAM encoder input for the crop (x0, y0, w, h) of img: resize
 * so the longest side is 1024 (PIL bilinear, as SamPredictor does),
 * normalize with the ImageNet mean/std SAM was trained with and zero-pad
 * to 3 x 1024 x 1024 CHW. Returns the resized extent in in_h/in_w. */
void image_encoder_input(const struct sam_image *img, int x0, int y0, int w,
                         int h, float *out, int *in_h, int *in_w);

void image_free(struct sam_image *img);

#endif /* I_SAM_OPENCL_IMAGE_H */
