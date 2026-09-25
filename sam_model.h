#ifndef I_SAM_OPENCL_SAM_MODEL_H
#define I_SAM_OPENCL_SAM_MODEL_H

#include "ocl_backend.h"
#include "pth_loader.h"

#define SAM_IMG 1024   /* Encoder input side. */
#define SAM_GRID 64    /* Embedding grid side (SAM_IMG / 16). */
#define SAM_DIM 256    /* Prompt/decoder embedding width. */
#define SAM_LOWRES 256 /* Low-resolution mask side. */

struct sam_attn_w {
    cl_mem qw, qb, kw, kb, vw, vb, ow, ob;
    int dint;
};

struct sam_block {
    cl_mem n1w, n1b, relh, relw, qkvw, qkvb, projw, projb, n2w, n2b;
    cl_mem l1w, l1b, l2w, l2b;
    int window; /* Window side, 0 for global attention. */
};

struct sam_dec_layer {
    struct sam_attn_w self, t2i, i2t;
    cl_mem n1w, n1b, n2w, n2b, n3w, n3b, n4w, n4b;
    cl_mem m1w, m1b, m2w, m2b;
};

struct sam_model {
    struct ocl_backend *ocl;
    const char *variant;
    int embed, depth, heads, hd, mlp;

    /* Image encoder. */
    cl_mem pe_w, pe_b, pos;
    struct sam_block *blocks;
    cl_mem neck0, neck1w, neck1b, neck2, neck3w, neck3b;

    /* Prompt encoder (host side, tiny). */
    float gauss[2 * 128];
    float point_emb[4][SAM_DIM];
    float not_a_point[SAM_DIM];
    float no_mask[SAM_DIM];

    /* Mask decoder. */
    struct sam_dec_layer layers[2];
    struct sam_attn_w final_t2i;
    cl_mem nfw, nfb;
    cl_mem up0w, up0b, up1w, up1b, up3w, up3b;
    float iou_token[SAM_DIM], mask_tokens[4 * SAM_DIM];
    float *hyper_w[4][3], *hyper_b[4][3];
    float *iou_w[3], *iou_b[3];

    cl_kernel k_gemm, k_gemm_big, k_attn_few, k_attn_many, k_ln, k_softmax,
        k_patch, k_add, k_wpart, k_wunpart, k_im2col, k_convt, k_logits, k_post;

    /* Encoder work buffers. */
    int head_chunk;
    cl_mem img, x, xn, xw, qkv, scores, attn, tmp, mlpbuf;
    cl_mem emb;

    /* Per-image decoder caches: src = embedding + no-mask embedding,
     * dense positional encoding, and the layer-0 projections that do
     * not depend on the prompt. */
    cl_mem src, pe_img, srcpe, kp0, vp0, qp0;

    /* Decoder batch buffers, sized for batch prompts. */
    int batch;
    cl_mem qpe, qry, qin, sq, sk, sv, so, sh;
    cl_mem keys, kin, bq, bk, bv, bo;
    cl_mem u1, g2, u2, hyper, lowres;
    float *h_queries;

    /* Postprocessed binary masks and their statistics. */
    cl_mem mask, sel, stats;
    size_t mask_cap;
};

/* Load weights from a SAM checkpoint (vit_h, vit_l or vit_b, detected
 * from tensor shapes) onto the device and allocate work buffers for
 * decoding batch prompts at a time. */
void sam_load(struct sam_model *m, struct ocl_backend *ocl,
              const struct pth_file *pf, int batch);

/* Encode a normalized, zero-padded 3 x 1024 x 1024 CHW image and prepare
 * the prompt-independent decoder caches. */
void sam_encode(struct sam_model *m, const float *img);

/* Copy the current 64 x 64 x 256 (token-major) embedding to host. */
void sam_read_embedding(struct sam_model *m, float *out);

/* Decode P <= batch single-point prompts, points given in encoder
 * input pixel coordinates (x, y pairs). Writes the three multimask IoU
 * predictions per prompt to iou[P * 3] and leaves the low-resolution
 * logits on the device. */
void sam_decode(struct sam_model *m, int P, const float *points, float *iou);

/* Postprocess n selected low-resolution masks (index p * 3 + k) to an
 * out_h x out_w crop whose resized encoder input extent is in_h x in_w.
 * stats receives 6 ints per mask: count(logit > offset), count(logit >
 * -offset), xmin, ymin, xmax, ymax (xmax < 0 if empty). */
void sam_postprocess(struct sam_model *m, int n, const int *sel, int in_h,
                     int in_w, int out_h, int out_w, float offset, int *stats);

/* Read rows y0..y1 and columns x0..x1 (inclusive) of postprocessed mask
 * k into dst (row-major, (x1 - x0 + 1) wide). */
void sam_read_mask(struct sam_model *m, int k, int out_h, int out_w, int x0,
                   int y0, int x1, int y1, unsigned char *dst);

void sam_free(struct sam_model *m);

#endif /* I_SAM_OPENCL_SAM_MODEL_H */
