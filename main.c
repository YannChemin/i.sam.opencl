/****************************************************************************
 *
 * MODULE:       i.sam.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Segment a 3-band imagery group with Meta's Segment
 *               Anything Model (SAM) on an OpenCL device, reading the
 *               original PyTorch checkpoint, and write the segments as a
 *               clean vector area map.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "amg.h"
#include "image.h"
#include "ocl_backend.h"
#include "pth_loader.h"
#include "sam_model.h"
#include "vectorize.h"

static double now(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* SAM variants in order of preference (quality first), with the width
 * of the image encoder that identifies them in a checkpoint. */
static const struct {
    const char *name;
    long embed;
} variants[] = {{"vit_h", 1280}, {"vit_l", 1024}, {"vit_b", 768}};

#define N_VARIANTS ((int)(sizeof(variants) / sizeof(variants[0])))

static int cmp_names(const void *a, const void *b)
{
    return strcmp(*(char *const *)a, *(char *const *)b);
}

/* Path of the checkpoint to use in a folder of downloaded weights: a .pth
 * file whose name contains the variant (e.g. sam_vit_h_4b8939.pth or a
 * renamed sam_vit_h_checkpoint.pth). With want == NULL the best variant
 * present is chosen. Names are scanned in sorted order, so the choice is
 * reproducible when a folder holds several files of one variant. */
static char *find_checkpoint(const char *dir, const char *want)
{
    DIR *d;
    struct dirent *e;
    char **names = NULL, path[GPATH_MAX];
    int n = 0, cap = 0, i, v, found = -1, other = 0;
    char *result = NULL;

    d = opendir(dir);
    if (!d)
        G_fatal_error(_("Unable to open weights folder <%s>"), dir);
    while ((e = readdir(d))) {
        size_t len = strlen(e->d_name);
        struct stat st;

        if (len < 5 || strcmp(e->d_name + len - 4, ".pth") != 0)
            continue;
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode))
            continue;
        if (n == cap) {
            cap = cap ? 2 * cap : 16;
            names = G_realloc(names, sizeof(char *) * cap);
        }
        names[n++] = G_store(e->d_name);
    }
    closedir(d);
    if (n)
        qsort(names, n, sizeof(char *), cmp_names);

    for (v = 0; v < N_VARIANTS && found < 0; v++) {
        if (want && strcmp(want, variants[v].name) != 0)
            continue;
        for (i = 0; i < n; i++) {
            if (!strstr(names[i], variants[v].name))
                continue;
            if (found < 0) {
                found = i;
                snprintf(path, sizeof(path), "%s/%s", dir, names[i]);
                result = G_store(path);
            }
            else
                other++;
        }
    }
    if (other)
        G_warning(_("Weights folder <%s> holds several checkpoints of that "
                    "model; using <%s>"),
                  dir, names[found]);
    for (i = 0; i < n; i++)
        G_free(names[i]);
    G_free(names);

    if (!result) {
        if (want)
            G_fatal_error(_("No %s checkpoint (a .pth file with '%s' in its "
                            "name) in weights folder <%s>"),
                          want, want, dir);
        G_fatal_error(_("No SAM checkpoint (a .pth file with vit_h, vit_l "
                        "or vit_b in its name) in weights folder <%s>"),
                      dir);
    }
    return result;
}

static void dump_masks(const char *dir, const struct mask_list *l)
{
    char path[GPATH_MAX];
    FILE *fp;
    int i;

    snprintf(path, sizeof(path), "%s/masks.csv", dir);
    fp = fopen(path, "w");
    if (!fp)
        G_fatal_error(_("Unable to write debug dump <%s>"), path);
    fprintf(fp, "x0,y0,x1,y1,area,predicted_iou,stability\n");
    for (i = 0; i < l->n; i++)
        fprintf(fp, "%d,%d,%d,%d,%ld,%.6f,%.6f\n", l->v[i].x0, l->v[i].y0,
                l->v[i].x1, l->v[i].y1, l->v[i].area, l->v[i].iou,
                l->v[i].stability);
    fclose(fp);
}

int main(int argc, char *argv[])
{
    struct GModule *module;
    struct {
        struct Option *group, *subgroup, *checkpoint, *weights_dir, *model,
            *output, *raster, *percentile, *points, *pred_iou, *stability,
            *stab_offset, *nms, *crop_layers, *crop_downscale, *minsize,
            *maxsize, *overlap, *simplify, *tile_size, *tile_overlap, *batch,
            *device, *platform;
    } opt;
    struct ocl_backend ocl;
    struct pth_file pf;
    struct sam_model model;
    struct sam_image img;
    struct mask_list masks;
    struct amg_params ap;
    struct vect_params vp;
    struct Cell_head win;
    double lo, hi, t0, t1;
    char source[GNAME_MAX * 2 + 32], *desc, *checkpoint;
    const struct pth_tensor *pos;
    int v;

    G_gisinit(argv[0]);

    module = G_define_module();
    G_add_keyword(_("imagery"));
    G_add_keyword(_("segmentation"));
    G_add_keyword(_("object recognition"));
    G_add_keyword(_("deep learning"));
    G_add_keyword(_("GPU"));
    G_add_keyword(_("OpenCL"));
    G_add_keyword(_("vectorization"));
    module->description =
        _("Segments a 3-band imagery group with the Segment Anything Model "
          "(SAM) on an OpenCL device and writes the segments as vector "
          "areas.");

    opt.group = G_define_standard_option(G_OPT_I_GROUP);
    opt.group->description =
        _("Name of input imagery group with exactly 3 raster maps (R, G, B)");

    opt.subgroup = G_define_standard_option(G_OPT_I_SUBGROUP);
    opt.subgroup->required = NO;
    opt.subgroup->description =
        _("Name of subgroup with exactly 3 raster maps to use instead of "
          "the whole group");

    opt.checkpoint = G_define_standard_option(G_OPT_F_INPUT);
    opt.checkpoint->key = "checkpoint";
    opt.checkpoint->required = NO;
    opt.checkpoint->label = _("SAM checkpoint file");
    opt.checkpoint->description =
        _("Original PyTorch checkpoint, e.g. sam_vit_h_4b8939.pth (vit_h, "
          "vit_l or vit_b)");
    opt.checkpoint->guisection = _("Model");

    opt.weights_dir = G_define_standard_option(G_OPT_M_DIR);
    opt.weights_dir->key = "weights_dir";
    opt.weights_dir->required = NO;
    opt.weights_dir->label = _("Folder of downloaded SAM checkpoints");
    opt.weights_dir->description =
        _("Folder holding one or more .pth checkpoints named with vit_h, "
          "vit_l or vit_b; used instead of checkpoint");
    opt.weights_dir->guisection = _("Model");

    opt.model = G_define_option();
    opt.model->key = "model";
    opt.model->type = TYPE_STRING;
    opt.model->required = NO;
    opt.model->options = "vit_h,vit_l,vit_b";
    opt.model->label = _("SAM model to use from the weights folder");
    opt.model->description =
        _("Default: the best one present (vit_h, then vit_l, then vit_b)");
    G_asprintf(&desc, "vit_h;%s;vit_l;%s;vit_b;%s",
               _("ViT-H, 2.4 GiB, best quality"), _("ViT-L, 1.2 GiB"),
               _("ViT-B, 358 MiB, fastest"));
    opt.model->descriptions = desc;
    opt.model->guisection = _("Model");

    opt.output = G_define_standard_option(G_OPT_V_OUTPUT);
    opt.output->description = _("Name for output vector map of segments");

    opt.raster = G_define_standard_option(G_OPT_R_OUTPUT);
    opt.raster->key = "raster";
    opt.raster->required = NO;
    opt.raster->description =
        _("Name for optional output raster map of segment categories");

    opt.percentile = G_define_option();
    opt.percentile->key = "percentile";
    opt.percentile->type = TYPE_DOUBLE;
    opt.percentile->key_desc = "lower,upper";
    opt.percentile->answer = "2,98";
    opt.percentile->options = "0-100";
    opt.percentile->description =
        _("Percentiles of each band stretched to the 0-255 range fed to "
          "SAM");
    opt.percentile->guisection = _("Input");

    opt.points = G_define_option();
    opt.points->key = "points_per_side";
    opt.points->type = TYPE_INTEGER;
    opt.points->answer = "32";
    opt.points->options = "1-128";
    opt.points->description =
        _("Number of prompt points along each side of a crop");
    opt.points->guisection = _("SAM");

    opt.pred_iou = G_define_option();
    opt.pred_iou->key = "iou_threshold";
    opt.pred_iou->type = TYPE_DOUBLE;
    opt.pred_iou->answer = "0.88";
    opt.pred_iou->options = "0-1";
    opt.pred_iou->description =
        _("Minimum mask quality predicted by the model");
    opt.pred_iou->guisection = _("SAM");

    opt.stability = G_define_option();
    opt.stability->key = "stability_threshold";
    opt.stability->type = TYPE_DOUBLE;
    opt.stability->answer = "0.95";
    opt.stability->options = "0-1";
    opt.stability->description =
        _("Minimum mask stability under a shift of the logit cutoff");
    opt.stability->guisection = _("SAM");

    opt.stab_offset = G_define_option();
    opt.stab_offset->key = "stability_offset";
    opt.stab_offset->type = TYPE_DOUBLE;
    opt.stab_offset->answer = "1.0";
    opt.stab_offset->description =
        _("Logit cutoff shift used to compute the stability score");
    opt.stab_offset->guisection = _("SAM");

    opt.nms = G_define_option();
    opt.nms->key = "nms_threshold";
    opt.nms->type = TYPE_DOUBLE;
    opt.nms->answer = "0.7";
    opt.nms->options = "0-1";
    opt.nms->description =
        _("Box IoU above which duplicate masks are suppressed");
    opt.nms->guisection = _("SAM");

    opt.crop_layers = G_define_option();
    opt.crop_layers->key = "crop_layers";
    opt.crop_layers->type = TYPE_INTEGER;
    opt.crop_layers->answer = "0";
    opt.crop_layers->options = "0-3";
    opt.crop_layers->description =
        _("Number of zoomed-in crop layers (layer n has 4^n crops)");
    opt.crop_layers->guisection = _("SAM");

    opt.crop_downscale = G_define_option();
    opt.crop_downscale->key = "crop_points_downscale";
    opt.crop_downscale->type = TYPE_INTEGER;
    opt.crop_downscale->answer = "1";
    opt.crop_downscale->options = "1-8";
    opt.crop_downscale->description =
        _("Divide points_per_side by this factor in each crop layer");
    opt.crop_downscale->guisection = _("SAM");

    opt.minsize = G_define_option();
    opt.minsize->key = "minsize";
    opt.minsize->type = TYPE_INTEGER;
    opt.minsize->answer = "100";
    opt.minsize->options = "0-";
    opt.minsize->label = _("Minimum segment size in cells");
    opt.minsize->description =
        _("Smaller holes and islands are removed from masks and smaller "
          "pieces merged into their neighbours");
    opt.minsize->guisection = _("Cleaning");

    opt.maxsize = G_define_option();
    opt.maxsize->key = "maxsize";
    opt.maxsize->type = TYPE_INTEGER;
    opt.maxsize->required = NO;
    opt.maxsize->options = "1-";
    opt.maxsize->label = _("Maximum mask size in cells");
    opt.maxsize->description =
        _("Larger masks, typically a background mask spanning the whole "
          "scene, are discarded");
    opt.maxsize->guisection = _("Cleaning");

    opt.overlap = G_define_option();
    opt.overlap->key = "overlap";
    opt.overlap->type = TYPE_STRING;
    opt.overlap->options = "small,large,score";
    opt.overlap->answer = "small";
    opt.overlap->description = _("Mask kept where masks overlap");
    G_asprintf(&desc, "small;%s;large;%s;score;%s",
               _("smallest mask (finest partition)"),
               _("largest mask (coarsest partition)"),
               _("mask with the highest predicted IoU"));
    opt.overlap->descriptions = desc;
    opt.overlap->guisection = _("Cleaning");

    opt.simplify = G_define_option();
    opt.simplify->key = "simplify";
    opt.simplify->type = TYPE_DOUBLE;
    opt.simplify->options = "0-";
    opt.simplify->label = _("Boundary simplification tolerance in map units");
    opt.simplify->description =
        _("Douglas-Peucker tolerance; default is one cell, 0 keeps the "
          "raster staircase");
    opt.simplify->guisection = _("Cleaning");

    opt.tile_size = G_define_option();
    opt.tile_size->key = "tile_size";
    opt.tile_size->type = TYPE_INTEGER;
    opt.tile_size->answer = "1024";
    opt.tile_size->options = "64-";
    opt.tile_size->description =
        _("Side of the tiles, in cells, a larger region is split into");
    opt.tile_size->guisection = _("Processing");

    opt.tile_overlap = G_define_option();
    opt.tile_overlap->key = "tile_overlap";
    opt.tile_overlap->type = TYPE_INTEGER;
    opt.tile_overlap->answer = "256";
    opt.tile_overlap->options = "0-";
    opt.tile_overlap->description = _("Overlap between tiles in cells");
    opt.tile_overlap->guisection = _("Processing");

    opt.batch = G_define_option();
    opt.batch->key = "batch";
    opt.batch->type = TYPE_INTEGER;
    opt.batch->answer = "32";
    opt.batch->options = "1-256";
    opt.batch->description =
        _("Number of point prompts decoded at once on the device");
    opt.batch->guisection = _("Processing");

    opt.device = G_define_option();
    opt.device->key = "device";
    opt.device->type = TYPE_STRING;
    opt.device->options = "auto,gpu,cpu";
    opt.device->answer = "auto";
    opt.device->description = _("OpenCL device type");
    opt.device->guisection = _("Processing");

    opt.platform = G_define_option();
    opt.platform->key = "platform";
    opt.platform->type = TYPE_STRING;
    opt.platform->required = NO;
    opt.platform->description =
        _("Use only OpenCL platforms whose name contains this text");
    opt.platform->guisection = _("Processing");

    G_option_required(opt.checkpoint, opt.weights_dir, NULL);
    G_option_exclusive(opt.checkpoint, opt.weights_dir, NULL);
    G_option_requires(opt.model, opt.weights_dir, NULL);

    if (G_parser(argc, argv))
        exit(EXIT_FAILURE);

    lo = atof(opt.percentile->answers[0]);
    hi = opt.percentile->answers[1] ? atof(opt.percentile->answers[1]) : -1;
    if (!(hi > lo))
        G_fatal_error(_("<%s> needs two increasing values"),
                      opt.percentile->key);

    ap.points_per_side = atoi(opt.points->answer);
    ap.pred_iou_thresh = atof(opt.pred_iou->answer);
    ap.stability_thresh = atof(opt.stability->answer);
    ap.stability_offset = atof(opt.stab_offset->answer);
    ap.box_nms_thresh = atof(opt.nms->answer);
    ap.crop_layers = atoi(opt.crop_layers->answer);
    ap.crop_nms_thresh = 0.7;
    ap.crop_overlap_ratio = 512.0 / 1500.0;
    ap.crop_points_downscale = atoi(opt.crop_downscale->answer);
    ap.min_region_area = atoi(opt.minsize->answer);
    ap.tile_size = atoi(opt.tile_size->answer);
    ap.tile_overlap = atoi(opt.tile_overlap->answer);
    ap.dump_dir = getenv("I_SAM_OPENCL_DUMP");
    if (ap.dump_dir && !*ap.dump_dir)
        ap.dump_dir = NULL;
    if (ap.tile_overlap * 2 > ap.tile_size)
        G_fatal_error(_("<%s> must not exceed half of <%s>"),
                      opt.tile_overlap->key, opt.tile_size->key);

    G_get_window(&win);
    vp.overlap = strcmp(opt.overlap->answer, "large") == 0   ? OVERLAP_LARGE
                 : strcmp(opt.overlap->answer, "score") == 0 ? OVERLAP_SCORE
                                                             : OVERLAP_SMALL;
    vp.minsize = ap.min_region_area;
    vp.maxsize = opt.maxsize->answer ? atol(opt.maxsize->answer) : 0;
    vp.simplify = opt.simplify->answer
                      ? atof(opt.simplify->answer)
                      : (win.ew_res > win.ns_res ? win.ew_res : win.ns_res);
    vp.vector_out = opt.output->answer;
    vp.raster_out = opt.raster->answer;
    snprintf(source, sizeof(source), "i.sam.opencl group=%s%s%s",
             opt.group->answer, opt.subgroup->answer ? " subgroup=" : "",
             opt.subgroup->answer ? opt.subgroup->answer : "");
    vp.source = source;

    /* Validate the cheap inputs (group, checkpoint index) before the
     * OpenCL device is initialized and the weights are uploaded. */
    image_read_group(opt.group->answer, opt.subgroup->answer, lo, hi, &img);
    checkpoint =
        opt.checkpoint->answer
            ? G_store(opt.checkpoint->answer)
            : find_checkpoint(opt.weights_dir->answer, opt.model->answer);
    if (opt.weights_dir->answer)
        G_message(_("Using checkpoint <%s>"), checkpoint);
    pth_open(&pf, checkpoint);
    pos = pth_find(&pf, "image_encoder.pos_embed");
    if (!pos || !pth_find(&pf, "mask_decoder.iou_token.weight"))
        G_fatal_error(_("<%s> is not a Segment Anything checkpoint"),
                      checkpoint);
    /* The file name chose the model; the tensor shapes must agree. */
    if (opt.model->answer) {
        for (v = 0; v < N_VARIANTS; v++)
            if (strcmp(opt.model->answer, variants[v].name) == 0)
                break;
        if (pos->shape[pos->ndim - 1] != variants[v].embed)
            G_fatal_error(_("<%s> is named as a %s checkpoint but does not "
                            "hold a %s model"),
                          checkpoint, opt.model->answer, opt.model->answer);
    }
    G_free(checkpoint);

    t0 = now();
    ocl_init(&ocl, opt.device->answer, opt.platform->answer);
    sam_load(&model, &ocl, &pf, atoi(opt.batch->answer));
    pth_close(&pf);
    t1 = now();
    G_message(_("Model ready in %.1f s"), t1 - t0);

    amg_generate(&model, &img, &ap, &masks);
    t0 = now();
    G_message(_("Segmentation took %.1f s"), t0 - t1);
    if (ap.dump_dir)
        dump_masks(ap.dump_dir, &masks);

    sam_free(&model);
    ocl_free(&ocl);

    vectorize_masks(&masks, &img, &vp);
    G_message(_("Vectorization took %.1f s"), now() - t0);

    mask_list_free(&masks);
    image_free(&img);

    exit(EXIT_SUCCESS);
}
