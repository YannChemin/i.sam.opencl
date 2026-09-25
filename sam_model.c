/****************************************************************************
 *
 * MODULE:       i.sam.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Segment Anything (SAM) image encoder and mask decoder
 *               forward passes on OpenCL, weights read directly from the
 *               original PyTorch checkpoint.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "pth_loader.h"
#include "sam_model.h"

#define ACT_NONE 0
#define ACT_GELU 1
#define ACT_RELU 2

#define ENC_EPS 1e-6f /* nn.LayerNorm(eps=1e-6) and LayerNorm2d. */
#define DEC_EPS 1e-5f /* nn.LayerNorm default, decoder transformer. */

#define NTOK 7 /* IoU token, 4 mask tokens, point, padding point. */
#define DEC_HEADS 8

/* Weight loading. */

static size_t loaded_bytes;

static cl_mem load_dev(struct sam_model *m, const struct pth_file *pf,
                       const char *name, int ndim, const long *shape)
{
    const struct pth_tensor *t = pth_require(pf, name, ndim, shape);
    float *h = pth_read_f32(pf, t);
    size_t n = pth_numel(t) * sizeof(float);
    cl_mem b = ocl_upload(m->ocl, h, n, name);

    loaded_bytes += n;
    G_free(h);
    return b;
}

static float *load_host(const struct pth_file *pf, const char *name, int ndim,
                        const long *shape)
{
    return pth_read_f32(pf, pth_require(pf, name, ndim, shape));
}

static void load_host_into(const struct pth_file *pf, const char *name,
                           int ndim, const long *shape, float *dst)
{
    const struct pth_tensor *t = pth_require(pf, name, ndim, shape);
    float *h = pth_read_f32(pf, t);

    memcpy(dst, h, pth_numel(t) * sizeof(float));
    G_free(h);
}

static cl_mem load_dev_fmt(struct sam_model *m, const struct pth_file *pf,
                           int ndim, const long *shape, const char *fmt, ...)
{
    char name[256];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    return load_dev(m, pf, name, ndim, shape);
}

static void load_attn(struct sam_model *m, const struct pth_file *pf,
                      const char *prefix, int dint, struct sam_attn_w *a)
{
    long w_in[2] = {dint, SAM_DIM}, w_out[2] = {SAM_DIM, dint};
    long b_in[1] = {dint}, b_out[1] = {SAM_DIM};

    a->dint = dint;
    a->qw = load_dev_fmt(m, pf, 2, w_in, "%s.q_proj.weight", prefix);
    a->qb = load_dev_fmt(m, pf, 1, b_in, "%s.q_proj.bias", prefix);
    a->kw = load_dev_fmt(m, pf, 2, w_in, "%s.k_proj.weight", prefix);
    a->kb = load_dev_fmt(m, pf, 1, b_in, "%s.k_proj.bias", prefix);
    a->vw = load_dev_fmt(m, pf, 2, w_in, "%s.v_proj.weight", prefix);
    a->vb = load_dev_fmt(m, pf, 1, b_in, "%s.v_proj.bias", prefix);
    a->ow = load_dev_fmt(m, pf, 2, w_out, "%s.out_proj.weight", prefix);
    a->ob = load_dev_fmt(m, pf, 1, b_out, "%s.out_proj.bias", prefix);
}

static void release_attn(struct sam_attn_w *a)
{
    ocl_release(&a->qw);
    ocl_release(&a->qb);
    ocl_release(&a->kw);
    ocl_release(&a->kb);
    ocl_release(&a->vw);
    ocl_release(&a->vb);
    ocl_release(&a->ow);
    ocl_release(&a->ob);
}

/* Kernel launch helpers. */

#define SETARG(k, i, v) ocl_check(clSetKernelArg(k, i, sizeof(v), &(v)), #k)

/* Accumulated kernel times when profiling is enabled. */
static struct {
    const char *what;
    double ms;
    long calls;
} prof[64];
static int nprof;

static void run(struct sam_model *m, cl_kernel k, int dim, const size_t *g,
                const size_t *l, const char *what)
{
    cl_event ev;
    cl_ulong t0, t1;
    int i;

    ocl_check(clEnqueueNDRangeKernel(m->ocl->queue, k, dim, NULL, g, l, 0, NULL,
                                     m->ocl->profile ? &ev : NULL),
              what);
    if (!m->ocl->profile)
        return;
    clWaitForEvents(1, &ev);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0,
                            NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1,
                            NULL);
    clReleaseEvent(ev);
    for (i = 0; i < nprof && strcmp(prof[i].what, what) != 0; i++)
        ;
    if (i == nprof && nprof < 64)
        prof[nprof++].what = what;
    if (i < 64) {
        prof[i].ms += (t1 - t0) * 1e-6;
        prof[i].calls++;
    }
}

static void run1d(struct sam_model *m, cl_kernel k, size_t n, const char *what)
{
    size_t l = 256, g = (n + l - 1) / l * l;

    run(m, k, 1, &g, &l, what);
}

/* One GEMM operand: buffer, element offset, leading dimension and the
 * two batch strides. */
struct opnd {
    cl_mem buf;
    cl_long off;
    cl_int ld;
    cl_long s0, s1;
};

static struct opnd mat(cl_mem buf, long off, int ld)
{
    struct opnd o = {buf, off, ld, 0, 0};

    return o;
}

static struct opnd bmat(cl_mem buf, long off, int ld, long s0, long s1)
{
    struct opnd o = {buf, off, ld, s0, s1};

    return o;
}

static void gemm(struct sam_model *m, int M, int N, int K, struct opnd A,
                 struct opnd B, int transb, struct opnd C, int nb, int nb1,
                 cl_mem bias, cl_mem R, long roff, float alpha, int act)
{
    cl_int cM = M, cN = N, cK = K, ct = transb, cnb1 = nb1, cact = act;
    cl_long croff = roff;
    cl_float calpha = alpha;
    size_t g[3], l[3] = {16, 16, 1};
    /* The 128 x 128 float4 kernel needs every A/B row start 16-byte
     * aligned; small or misaligned products use the generic kernel. */
    int big = M >= 128 && N >= 64 && K % 4 == 0 && A.ld % 4 == 0 &&
              A.off % 4 == 0 && A.s0 % 4 == 0 && A.s1 % 4 == 0 &&
              B.ld % 4 == 0 && B.off % 4 == 0 && B.s0 % 4 == 0 && B.s1 % 4 == 0;
    int tile = big ? 128 : 64;
    cl_kernel k = big ? m->k_gemm_big : m->k_gemm;

    SETARG(k, 0, cM);
    SETARG(k, 1, cN);
    SETARG(k, 2, cK);
    SETARG(k, 3, A.buf);
    SETARG(k, 4, A.off);
    SETARG(k, 5, A.ld);
    SETARG(k, 6, A.s0);
    SETARG(k, 7, A.s1);
    SETARG(k, 8, B.buf);
    SETARG(k, 9, B.off);
    SETARG(k, 10, B.ld);
    SETARG(k, 11, B.s0);
    SETARG(k, 12, B.s1);
    SETARG(k, 13, ct);
    SETARG(k, 14, C.buf);
    SETARG(k, 15, C.off);
    SETARG(k, 16, C.ld);
    SETARG(k, 17, C.s0);
    SETARG(k, 18, C.s1);
    SETARG(k, 19, cnb1);
    SETARG(k, 20, bias);
    SETARG(k, 21, R);
    SETARG(k, 22, croff);
    SETARG(k, 23, calpha);
    SETARG(k, 24, cact);
    g[0] = (size_t)((N + tile - 1) / tile) * 16;
    g[1] = (size_t)((M + tile - 1) / tile) * 16;
    g[2] = nb;
    run(m, k, 3, g, l, big ? "gemm_big" : "gemm");
}

/* Y[rows, out] = act(X[rows, in] W^T + b) (+ Y when residual). */
static void linear(struct sam_model *m, int rows, int in, int out, cl_mem X,
                   long xoff, cl_mem W, cl_mem b, cl_mem Y, long yoff, int act,
                   int residual)
{
    gemm(m, rows, out, in, mat(X, xoff, in), mat(W, 0, in), 1,
         mat(Y, yoff, out), 1, 1, b, residual ? Y : NULL, yoff, 1.0f, act);
}

/* Row LayerNorm: 256 items per row for the encoder's wide rows, one
 * 64-item wavefront per row for the decoder's 64/256-channel rows. */
static void layernorm(struct sam_model *m, long rows, int C, cl_mem X, cl_mem Y,
                      cl_mem g, cl_mem b, float eps, int act)
{
    cl_kernel k = m->k_ln;
    cl_int cC = C, cact = act;
    cl_float ceps = eps;
    size_t ls = C >= 1024 ? 256 : 64, gs = rows * ls;

    SETARG(k, 0, X);
    SETARG(k, 1, Y);
    SETARG(k, 2, g);
    SETARG(k, 3, b);
    SETARG(k, 4, cC);
    SETARG(k, 5, ceps);
    SETARG(k, 6, cact);
    run(m, k, 1, &gs, &ls, "layernorm");
}

static void softmax_relpos(struct sam_model *m, cl_mem S, long rows, int L,
                           cl_mem Q, int ldq, int hd, int Nq, int hc, int h0,
                           int side, cl_mem Rh, cl_mem Rw)
{
    cl_kernel k = m->k_softmax;
    cl_int cL = L, cldq = ldq, chd = hd, cNq = Nq, chc = hc, ch0 = h0,
           cside = side;
    size_t gs = rows * 256, ls = 256;

    SETARG(k, 0, S);
    SETARG(k, 1, cL);
    SETARG(k, 2, Q);
    SETARG(k, 3, cldq);
    SETARG(k, 4, chd);
    SETARG(k, 5, cNq);
    SETARG(k, 6, chc);
    SETARG(k, 7, ch0);
    SETARG(k, 8, cside);
    SETARG(k, 9, Rh);
    SETARG(k, 10, Rw);
    run(m, k, 1, &gs, &ls, "softmax_relpos");
}

static void add_bcast(struct sam_model *m, cl_mem out, cl_mem a, cl_mem b,
                      long n, long bmod)
{
    cl_kernel k = m->k_add;
    cl_int cn = (cl_int)n, cb = (cl_int)bmod;

    SETARG(k, 0, out);
    SETARG(k, 1, a);
    SETARG(k, 2, b);
    SETARG(k, 3, cn);
    SETARG(k, 4, cb);
    run1d(m, k, n, "add_bcast");
}

static void convt_scatter(struct sam_model *m, cl_mem G, cl_mem out,
                          cl_mem bias, int P, int H, int W, int Co, int act)
{
    cl_kernel k = m->k_convt;
    cl_int cH = H, cW = W, cCo = Co, cact = act;
    cl_int total = P * 4 * H * W * Co;

    SETARG(k, 0, G);
    SETARG(k, 1, out);
    SETARG(k, 2, bias);
    SETARG(k, 3, cH);
    SETARG(k, 4, cW);
    SETARG(k, 5, cCo);
    SETARG(k, 6, total);
    SETARG(k, 7, cact);
    run1d(m, k, total, "convt_scatter");
}

/* Multi-head attention core on already projected tensors (decoder, 8
 * heads). qp/kp/vp hold P prompts of Nq/Nk rows of dint channels with
 * per-prompt strides sq/sk/sv (0 = shared by all prompts); the
 * recombined heads are written to o as P * Nq rows of dint. Scores,
 * softmax and weighting are fused: per query for the 7 prompt-token
 * keys, per work-group for the 4096 image-token keys. */
static void mha(struct sam_model *m, int P, int Nq, int Nk, int dint, cl_mem qp,
                long sq, cl_mem kp, long sk, cl_mem vp, long sv, cl_mem o)
{
    cl_int c = dint / DEC_HEADS, cNq = Nq, cNk = Nk, cd = dint;
    cl_int csq = (cl_int)sq, csk = (cl_int)sk, csv = (cl_int)sv;
    cl_int total = P * DEC_HEADS * Nq;
    cl_float scale = 1.0f / sqrtf((float)c);
    int many = Nk > 64;
    cl_kernel k = many ? m->k_attn_many : m->k_attn_few;

    if (many && (c != 16 || Nk > 4096))
        G_fatal_error("mha: unsupported shape");
    SETARG(k, 0, qp);
    SETARG(k, 1, csq);
    SETARG(k, 2, kp);
    SETARG(k, 3, csk);
    SETARG(k, 4, vp);
    SETARG(k, 5, csv);
    SETARG(k, 6, o);
    SETARG(k, 7, cNq);
    SETARG(k, 8, cNk);
    SETARG(k, 9, cd);
    SETARG(k, 10, c);
    SETARG(k, 11, scale);
    SETARG(k, 12, total);
    if (many) {
        size_t gs = (size_t)total * 256, ls = 256;

        run(m, k, 1, &gs, &ls, "attn_manykeys");
    }
    else
        run1d(m, k, total, "attn_fewkeys");
}

/* Model loading. */

static void load_encoder(struct sam_model *m, const struct pth_file *pf)
{
    const struct pth_tensor *t;
    long E, s1[1], s2[2], s4[4];
    char name[128];
    int i;

    t = pth_require(pf, "image_encoder.pos_embed", 4,
                    (long[]){1, SAM_GRID, SAM_GRID, -1});
    E = t->shape[3];
    m->embed = (int)E;
    for (m->depth = 0;; m->depth++) {
        snprintf(name, sizeof(name), "image_encoder.blocks.%d.norm1.weight",
                 m->depth);
        if (!pth_find(pf, name))
            break;
    }
    t = pth_require(pf, "image_encoder.blocks.0.attn.rel_pos_h", 2, NULL);
    m->hd = (int)t->shape[1];
    if (m->hd <= 0 || m->hd > 128 || E % m->hd != 0)
        G_fatal_error(_("Unsupported attention head width %d"), m->hd);
    m->heads = (int)(E / m->hd);
    t = pth_require(pf, "image_encoder.blocks.0.mlp.lin1.weight", 2,
                    (long[]){-1, E});
    m->mlp = (int)t->shape[0];

    m->variant = E == 1280   ? "vit_h"
                 : E == 1024 ? "vit_l"
                 : E == 768  ? "vit_b"
                             : "custom";
    G_message(_("SAM %s image encoder: width %d, depth %d, %d heads"),
              m->variant, m->embed, m->depth, m->heads);

    m->pe_w = load_dev(m, pf, "image_encoder.patch_embed.proj.weight", 4,
                       (long[]){E, 3, 16, 16});
    m->pe_b =
        load_dev(m, pf, "image_encoder.patch_embed.proj.bias", 1, (long[]){E});
    m->pos = load_dev(m, pf, "image_encoder.pos_embed", 4,
                      (long[]){1, SAM_GRID, SAM_GRID, E});

    m->blocks = G_calloc(m->depth, sizeof(struct sam_block));
    for (i = 0; i < m->depth; i++) {
        struct sam_block *b = &m->blocks[i];
        const char *p = "image_encoder.blocks";
        long rows;

        G_percent(i, m->depth, 5);
        snprintf(name, sizeof(name), "%s.%d.attn.rel_pos_h", p, i);
        t = pth_require(pf, name, 2, (long[]){-1, m->hd});
        rows = t->shape[0];
        if (rows % 2 == 0 || (rows + 1) / 2 > 128)
            G_fatal_error(_("Unsupported relative position table <%s>"), name);
        b->window = rows == 2 * SAM_GRID - 1 ? 0 : (int)(rows + 1) / 2;

        s1[0] = E;
        b->n1w = load_dev_fmt(m, pf, 1, s1, "%s.%d.norm1.weight", p, i);
        b->n1b = load_dev_fmt(m, pf, 1, s1, "%s.%d.norm1.bias", p, i);
        b->n2w = load_dev_fmt(m, pf, 1, s1, "%s.%d.norm2.weight", p, i);
        b->n2b = load_dev_fmt(m, pf, 1, s1, "%s.%d.norm2.bias", p, i);
        s2[0] = rows;
        s2[1] = m->hd;
        b->relh = load_dev_fmt(m, pf, 2, s2, "%s.%d.attn.rel_pos_h", p, i);
        b->relw = load_dev_fmt(m, pf, 2, s2, "%s.%d.attn.rel_pos_w", p, i);
        s2[0] = 3 * E;
        s2[1] = E;
        b->qkvw = load_dev_fmt(m, pf, 2, s2, "%s.%d.attn.qkv.weight", p, i);
        s1[0] = 3 * E;
        b->qkvb = load_dev_fmt(m, pf, 1, s1, "%s.%d.attn.qkv.bias", p, i);
        s2[0] = E;
        b->projw = load_dev_fmt(m, pf, 2, s2, "%s.%d.attn.proj.weight", p, i);
        s1[0] = E;
        b->projb = load_dev_fmt(m, pf, 1, s1, "%s.%d.attn.proj.bias", p, i);
        s2[0] = m->mlp;
        s2[1] = E;
        b->l1w = load_dev_fmt(m, pf, 2, s2, "%s.%d.mlp.lin1.weight", p, i);
        s1[0] = m->mlp;
        b->l1b = load_dev_fmt(m, pf, 1, s1, "%s.%d.mlp.lin1.bias", p, i);
        s2[0] = E;
        s2[1] = m->mlp;
        b->l2w = load_dev_fmt(m, pf, 2, s2, "%s.%d.mlp.lin2.weight", p, i);
        s1[0] = E;
        b->l2b = load_dev_fmt(m, pf, 1, s1, "%s.%d.mlp.lin2.bias", p, i);
    }
    G_percent(1, 1, 1);

    s4[0] = SAM_DIM;
    s4[1] = E;
    s4[2] = 1;
    s4[3] = 1;
    m->neck0 = load_dev(m, pf, "image_encoder.neck.0.weight", 4, s4);
    s1[0] = SAM_DIM;
    m->neck1w = load_dev(m, pf, "image_encoder.neck.1.weight", 1, s1);
    m->neck1b = load_dev(m, pf, "image_encoder.neck.1.bias", 1, s1);
    s4[1] = SAM_DIM;
    s4[2] = 3;
    s4[3] = 3;
    m->neck2 = load_dev(m, pf, "image_encoder.neck.2.weight", 4, s4);
    m->neck3w = load_dev(m, pf, "image_encoder.neck.3.weight", 1, s1);
    m->neck3b = load_dev(m, pf, "image_encoder.neck.3.bias", 1, s1);
}

static void load_decoder(struct sam_model *m, const struct pth_file *pf)
{
    const long v256[1] = {SAM_DIM}, e1[2] = {1, SAM_DIM};
    char name[160];
    int i, j;

    load_host_into(pf,
                   "prompt_encoder.pe_layer.positional_encoding_"
                   "gaussian_matrix",
                   2, (long[]){2, SAM_DIM / 2}, m->gauss);
    for (i = 0; i < 4; i++) {
        snprintf(name, sizeof(name),
                 "prompt_encoder.point_embeddings.%d.weight", i);
        load_host_into(pf, name, 2, e1, m->point_emb[i]);
    }
    load_host_into(pf, "prompt_encoder.not_a_point_embed.weight", 2, e1,
                   m->not_a_point);
    load_host_into(pf, "prompt_encoder.no_mask_embed.weight", 2, e1,
                   m->no_mask);
    load_host_into(pf, "mask_decoder.iou_token.weight", 2, e1, m->iou_token);
    load_host_into(pf, "mask_decoder.mask_tokens.weight", 2,
                   (long[]){4, SAM_DIM}, m->mask_tokens);

    for (i = 0; i < 2; i++) {
        struct sam_dec_layer *L = &m->layers[i];
        const char *p = "mask_decoder.transformer.layers";

        snprintf(name, sizeof(name), "%s.%d.self_attn", p, i);
        load_attn(m, pf, name, SAM_DIM, &L->self);
        snprintf(name, sizeof(name), "%s.%d.cross_attn_token_to_image", p, i);
        load_attn(m, pf, name, SAM_DIM / 2, &L->t2i);
        snprintf(name, sizeof(name), "%s.%d.cross_attn_image_to_token", p, i);
        load_attn(m, pf, name, SAM_DIM / 2, &L->i2t);
        L->n1w = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm1.weight", p, i);
        L->n1b = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm1.bias", p, i);
        L->n2w = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm2.weight", p, i);
        L->n2b = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm2.bias", p, i);
        L->n3w = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm3.weight", p, i);
        L->n3b = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm3.bias", p, i);
        L->n4w = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm4.weight", p, i);
        L->n4b = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm4.bias", p, i);
        L->m1w = load_dev_fmt(m, pf, 2, (long[]){2048, SAM_DIM},
                              "%s.%d.mlp.lin1.weight", p, i);
        L->m1b =
            load_dev_fmt(m, pf, 1, (long[]){2048}, "%s.%d.mlp.lin1.bias", p, i);
        L->m2w = load_dev_fmt(m, pf, 2, (long[]){SAM_DIM, 2048},
                              "%s.%d.mlp.lin2.weight", p, i);
        L->m2b = load_dev_fmt(m, pf, 1, v256, "%s.%d.mlp.lin2.bias", p, i);
    }
    load_attn(m, pf, "mask_decoder.transformer.final_attn_token_to_image",
              SAM_DIM / 2, &m->final_t2i);
    m->nfw = load_dev(m, pf, "mask_decoder.transformer.norm_final_attn.weight",
                      1, v256);
    m->nfb = load_dev(m, pf, "mask_decoder.transformer.norm_final_attn.bias", 1,
                      v256);

    m->up0w = load_dev(m, pf, "mask_decoder.output_upscaling.0.weight", 4,
                       (long[]){SAM_DIM, 64, 2, 2});
    m->up0b = load_dev(m, pf, "mask_decoder.output_upscaling.0.bias", 1,
                       (long[]){64});
    m->up1w = load_dev(m, pf, "mask_decoder.output_upscaling.1.weight", 1,
                       (long[]){64});
    m->up1b = load_dev(m, pf, "mask_decoder.output_upscaling.1.bias", 1,
                       (long[]){64});
    m->up3w = load_dev(m, pf, "mask_decoder.output_upscaling.3.weight", 4,
                       (long[]){64, 32, 2, 2});
    m->up3b = load_dev(m, pf, "mask_decoder.output_upscaling.3.bias", 1,
                       (long[]){32});

    for (i = 0; i < 4; i++) {
        for (j = 0; j < 3; j++) {
            long ws[2] = {j == 2 ? 32 : SAM_DIM, SAM_DIM};
            long bs[1] = {ws[0]};

            snprintf(name, sizeof(name),
                     "mask_decoder.output_hypernetworks_mlps.%d.layers.%d."
                     "weight",
                     i, j);
            m->hyper_w[i][j] = load_host(pf, name, 2, ws);
            snprintf(name, sizeof(name),
                     "mask_decoder.output_hypernetworks_mlps.%d.layers.%d.bias",
                     i, j);
            m->hyper_b[i][j] = load_host(pf, name, 1, bs);
        }
    }
    for (j = 0; j < 3; j++) {
        long ws[2] = {j == 2 ? 4 : SAM_DIM, SAM_DIM};
        long bs[1] = {ws[0]};

        snprintf(name, sizeof(name),
                 "mask_decoder.iou_prediction_head.layers.%d.weight", j);
        m->iou_w[j] = load_host(pf, name, 2, ws);
        snprintf(name, sizeof(name),
                 "mask_decoder.iou_prediction_head.layers.%d.bias", j);
        m->iou_b[j] = load_host(pf, name, 1, bs);
    }
}

/* SAM PositionEmbeddingRandom encoding of normalized (x, y) in [0, 1]. */
static void pe_encode(const struct sam_model *m, float x, float y, float *out)
{
    int i;

    x = 2.0f * x - 1.0f;
    y = 2.0f * y - 1.0f;
    for (i = 0; i < SAM_DIM / 2; i++) {
        float v = x * m->gauss[i] + y * m->gauss[SAM_DIM / 2 + i];

        v *= 2.0f * (float)M_PI;
        out[i] = sinf(v);
        out[SAM_DIM / 2 + i] = cosf(v);
    }
}

static void alloc_buffers(struct sam_model *m)
{
    struct ocl_backend *ocl = m->ocl;
    const size_t f = sizeof(float);
    const long T = SAM_GRID * SAM_GRID, E = m->embed;
    const long P = m->batch;
    long tw = T, scores = 0, colmax;
    size_t cap;
    float *pe;
    int i, y, x;

    /* Largest padded window partition and attention score matrix. */
    cap = ocl->max_alloc < ((size_t)512 << 20) ? ocl->max_alloc
                                               : ((size_t)512 << 20);
    m->head_chunk = (int)(cap / ((size_t)T * T * f));
    if (m->head_chunk < 1)
        G_fatal_error(_("OpenCL device cannot hold one global attention "
                        "matrix (%lu MiB)"),
                      (unsigned long)((size_t)T * T * f >> 20));
    if (m->head_chunk > m->heads)
        m->head_chunk = m->heads;
    for (i = 0; i < m->depth; i++) {
        int ws = m->blocks[i].window;
        long n, nw, s;

        if (ws) {
            nw = (SAM_GRID + ws - 1) / ws;
            n = nw * nw * ws * ws;
            s = nw * nw * m->heads * (long)ws * ws * ws * ws;
            if (n > tw)
                tw = n;
        }
        else
            s = (long)m->head_chunk * T * T;
        if (s > scores)
            scores = s;
    }

    m->img = ocl_alloc(ocl, 3L * SAM_IMG * SAM_IMG * f, "image");
    m->x = ocl_alloc(ocl, T * E * f, "tokens");
    m->xn = ocl_alloc(ocl, T * E * f, "normalized tokens");
    colmax = tw * E > T * 768 ? tw * E : T * 768;
    m->xw = ocl_alloc(ocl, colmax * f, "window partition");
    m->qkv = ocl_alloc(ocl, tw * 3 * E * f, "qkv");
    m->scores = ocl_alloc(ocl, scores * f, "attention scores");
    m->attn = ocl_alloc(ocl, tw * E * f, "attention output");
    m->tmp = ocl_alloc(ocl, tw * E * f, "projection");
    colmax = T * m->mlp > T * SAM_DIM * 9 ? T * m->mlp : T * SAM_DIM * 9;
    m->mlpbuf = ocl_alloc(ocl, colmax * f, "mlp");
    m->emb = ocl_alloc(ocl, T * SAM_DIM * f, "embedding");

    m->src = ocl_alloc(ocl, T * SAM_DIM * f, "decoder src");
    m->srcpe = ocl_alloc(ocl, T * SAM_DIM * f, "decoder src+pe");
    m->kp0 = ocl_alloc(ocl, T * 128 * f, "decoder k0");
    m->vp0 = ocl_alloc(ocl, T * 128 * f, "decoder v0");
    m->qp0 = ocl_alloc(ocl, T * 128 * f, "decoder q0");

    /* Dense positional encoding of the 64 x 64 embedding grid. */
    pe = G_malloc(T * SAM_DIM * f);
    for (y = 0; y < SAM_GRID; y++)
        for (x = 0; x < SAM_GRID; x++)
            pe_encode(m, (x + 0.5f) / SAM_GRID, (y + 0.5f) / SAM_GRID,
                      pe + ((long)y * SAM_GRID + x) * SAM_DIM);
    m->pe_img = ocl_upload(ocl, pe, T * SAM_DIM * f, "decoder pe");
    G_free(pe);

    m->qpe = ocl_alloc(ocl, P * NTOK * SAM_DIM * f, "query pe");
    m->qry = ocl_alloc(ocl, P * NTOK * SAM_DIM * f, "queries");
    m->qin = ocl_alloc(ocl, P * NTOK * SAM_DIM * f, "queries+pe");
    m->sq = ocl_alloc(ocl, P * NTOK * SAM_DIM * f, "token q");
    m->sk = ocl_alloc(ocl, P * NTOK * SAM_DIM * f, "token k");
    m->sv = ocl_alloc(ocl, P * NTOK * SAM_DIM * f, "token v");
    m->so = ocl_alloc(ocl, P * NTOK * SAM_DIM * f, "token attention");
    m->sh = ocl_alloc(ocl, P * NTOK * 2048 * f, "token mlp");
    m->keys = ocl_alloc(ocl, P * T * SAM_DIM * f, "keys");
    m->kin = ocl_alloc(ocl, P * T * SAM_DIM * f, "keys+pe");
    m->bq = ocl_alloc(ocl, P * T * 128 * f, "image q");
    m->bk = ocl_alloc(ocl, P * T * 128 * f, "image k");
    m->bv = ocl_alloc(ocl, P * T * 128 * f, "image v");
    m->bo = ocl_alloc(ocl, P * T * 128 * f, "image attention");
    m->u1 = ocl_alloc(ocl, P * 4 * T * 64 * f, "upscaling 1");
    m->g2 = ocl_alloc(ocl, P * 4 * T * 128 * f, "upscaling 2 gemm");
    m->u2 = ocl_alloc(ocl, P * 16 * T * 32 * f, "upscaling 2");
    m->hyper = ocl_alloc(ocl, P * 3 * 32 * f, "hypernetwork");
    m->lowres =
        ocl_alloc(ocl, P * 3 * SAM_LOWRES * SAM_LOWRES * f, "low-res masks");
    m->h_queries = G_malloc(P * NTOK * SAM_DIM * f);
    m->sel = ocl_alloc(ocl, P * 3 * sizeof(cl_int), "mask selection");
    m->stats = ocl_alloc(ocl, P * 3 * 6 * sizeof(cl_int), "mask statistics");
}

void sam_load(struct sam_model *m, struct ocl_backend *ocl,
              const struct pth_file *pf, int batch)
{
    memset(m, 0, sizeof(*m));
    m->ocl = ocl;
    m->batch = batch;

    m->k_gemm = ocl_kernel(ocl, "gemm");
    m->k_gemm_big = ocl_kernel(ocl, "gemm_big");
    m->k_ln = ocl_kernel(ocl, "layernorm");
    m->k_softmax = ocl_kernel(ocl, "softmax_relpos");
    m->k_attn_few = ocl_kernel(ocl, "attn_fewkeys");
    m->k_attn_many = ocl_kernel(ocl, "attn_manykeys");
    m->k_patch = ocl_kernel(ocl, "patch_im2col");
    m->k_add = ocl_kernel(ocl, "add_bcast");
    m->k_wpart = ocl_kernel(ocl, "window_partition");
    m->k_wunpart = ocl_kernel(ocl, "window_unpartition_add");
    m->k_im2col = ocl_kernel(ocl, "im2col3x3");
    m->k_convt = ocl_kernel(ocl, "convt_scatter");
    m->k_logits = ocl_kernel(ocl, "mask_logits");
    m->k_post = ocl_kernel(ocl, "mask_post");

    G_message(_("Loading SAM checkpoint <%s>..."), pf->path);
    loaded_bytes = 0;
    load_encoder(m, pf);
    load_decoder(m, pf);
    G_verbose_message(_("Uploaded %lu MiB of weights"),
                      (unsigned long)(loaded_bytes >> 20));

    alloc_buffers(m);
}

/* Image encoder forward pass. */

void sam_encode(struct sam_model *m, const float *img)
{
    const int T = SAM_GRID * SAM_GRID, E = m->embed, hd = m->hd;
    const float scale = 1.0f / sqrtf((float)hd);
    cl_kernel k;
    int i;

    ocl_write(m->ocl, m->img, 0, img, 3L * SAM_IMG * SAM_IMG * sizeof(float));

    /* Patch embedding and absolute position embedding. */
    {
        cl_int side = SAM_IMG, grid = SAM_GRID;

        k = m->k_patch;
        SETARG(k, 0, m->img);
        SETARG(k, 1, m->xw);
        SETARG(k, 2, side);
        SETARG(k, 3, grid);
        run1d(m, k, (size_t)T * 768, "patch_im2col");
        linear(m, T, 768, E, m->xw, 0, m->pe_w, m->pe_b, m->x, 0, ACT_NONE, 0);
        add_bcast(m, m->x, m->x, m->pos, (long)T * E, (long)T * E);
    }

    for (i = 0; i < m->depth; i++) {
        const struct sam_block *b = &m->blocks[i];
        const int ws = b->window;
        int nwx = 1, nwin = 1, Nq = T, rows = T, side = SAM_GRID, hc, h0;
        cl_mem A = m->xn;

        G_percent(i, m->depth, 10);
        layernorm(m, T, E, m->x, m->xn, b->n1w, b->n1b, ENC_EPS, ACT_NONE);

        if (ws) {
            cl_int cH = SAM_GRID, cW = SAM_GRID, cC = E, cws = ws, cnwx;
            cl_int total;

            nwx = (SAM_GRID + ws - 1) / ws;
            nwin = nwx * nwx;
            Nq = ws * ws;
            rows = nwin * Nq;
            side = ws;
            cnwx = nwx;
            total = rows * E;
            k = m->k_wpart;
            SETARG(k, 0, m->xn);
            SETARG(k, 1, m->xw);
            SETARG(k, 2, cH);
            SETARG(k, 3, cW);
            SETARG(k, 4, cC);
            SETARG(k, 5, cws);
            SETARG(k, 6, cnwx);
            SETARG(k, 7, total);
            run1d(m, k, total, "window_partition");
            A = m->xw;
        }

        linear(m, rows, E, 3 * E, A, 0, b->qkvw, b->qkvb, m->qkv, 0, ACT_NONE,
               0);

        hc = ws ? m->heads : m->head_chunk;
        for (h0 = 0; h0 < m->heads; h0 += hc) {
            int hcc = m->heads - h0 < hc ? m->heads - h0 : hc;
            long plane = (long)Nq * Nq, tstride = (long)Nq * 3 * E;

            gemm(m, Nq, Nq, hd, bmat(m->qkv, (long)h0 * hd, 3 * E, tstride, hd),
                 bmat(m->qkv, E + (long)h0 * hd, 3 * E, tstride, hd), 1,
                 bmat(m->scores, 0, Nq, hcc * plane, plane), nwin * hcc, hcc,
                 NULL, NULL, 0, scale, ACT_NONE);
            softmax_relpos(m, m->scores, (long)nwin * hcc * Nq, Nq, m->qkv,
                           3 * E, hd, Nq, hcc, h0, side, b->relh, b->relw);
            gemm(m, Nq, hd, Nq, bmat(m->scores, 0, Nq, hcc * plane, plane),
                 bmat(m->qkv, 2L * E + (long)h0 * hd, 3 * E, tstride, hd), 0,
                 bmat(m->attn, (long)h0 * hd, E, (long)Nq * E, hd), nwin * hcc,
                 hcc, NULL, NULL, 0, 1.0f, ACT_NONE);
        }

        if (ws) {
            cl_int cH = SAM_GRID, cW = SAM_GRID, cC = E, cws = ws, cnwx = nwx;

            linear(m, rows, E, E, m->attn, 0, b->projw, b->projb, m->tmp, 0,
                   ACT_NONE, 0);
            k = m->k_wunpart;
            SETARG(k, 0, m->x);
            SETARG(k, 1, m->tmp);
            SETARG(k, 2, cH);
            SETARG(k, 3, cW);
            SETARG(k, 4, cC);
            SETARG(k, 5, cws);
            SETARG(k, 6, cnwx);
            run1d(m, k, (size_t)T * E, "window_unpartition_add");
        }
        else
            linear(m, T, E, E, m->attn, 0, b->projw, b->projb, m->x, 0,
                   ACT_NONE, 1);

        layernorm(m, T, E, m->x, m->xn, b->n2w, b->n2b, ENC_EPS, ACT_NONE);
        linear(m, T, E, m->mlp, m->xn, 0, b->l1w, b->l1b, m->mlpbuf, 0,
               ACT_GELU, 0);
        linear(m, T, m->mlp, E, m->mlpbuf, 0, b->l2w, b->l2b, m->x, 0, ACT_NONE,
               1);
    }
    G_percent(1, 1, 1);

    /* Neck: 1x1 conv, LayerNorm2d, 3x3 conv, LayerNorm2d. */
    {
        cl_int cH = SAM_GRID, cW = SAM_GRID, cC = SAM_DIM;

        linear(m, T, E, SAM_DIM, m->x, 0, m->neck0, NULL, m->xn, 0, ACT_NONE,
               0);
        layernorm(m, T, SAM_DIM, m->xn, m->xn, m->neck1w, m->neck1b, ENC_EPS,
                  ACT_NONE);
        k = m->k_im2col;
        SETARG(k, 0, m->xn);
        SETARG(k, 1, m->mlpbuf);
        SETARG(k, 2, cH);
        SETARG(k, 3, cW);
        SETARG(k, 4, cC);
        run1d(m, k, (size_t)T * SAM_DIM * 9, "im2col3x3");
        linear(m, T, SAM_DIM * 9, SAM_DIM, m->mlpbuf, 0, m->neck2, NULL, m->emb,
               0, ACT_NONE, 0);
        layernorm(m, T, SAM_DIM, m->emb, m->emb, m->neck3w, m->neck3b, ENC_EPS,
                  ACT_NONE);
    }

    /* Prompt-independent decoder inputs. Without a mask prompt the
     * dense embedding is no_mask_embed broadcast over the grid. */
    {
        cl_mem nm = ocl_upload(m->ocl, m->no_mask, SAM_DIM * sizeof(float),
                               "no-mask embedding");
        const struct sam_dec_layer *L0 = &m->layers[0];

        add_bcast(m, m->src, m->emb, nm, (long)T * SAM_DIM, SAM_DIM);
        add_bcast(m, m->srcpe, m->src, m->pe_img, (long)T * SAM_DIM,
                  (long)T * SAM_DIM);
        linear(m, T, SAM_DIM, 128, m->srcpe, 0, L0->t2i.kw, L0->t2i.kb, m->kp0,
               0, ACT_NONE, 0);
        linear(m, T, SAM_DIM, 128, m->src, 0, L0->t2i.vw, L0->t2i.vb, m->vp0, 0,
               ACT_NONE, 0);
        linear(m, T, SAM_DIM, 128, m->srcpe, 0, L0->i2t.qw, L0->i2t.qb, m->qp0,
               0, ACT_NONE, 0);
        clFinish(m->ocl->queue);
        clReleaseMemObject(nm);
    }
}

void sam_read_embedding(struct sam_model *m, float *out)
{
    ocl_read(m->ocl, m->emb, 0, out,
             (size_t)SAM_GRID * SAM_GRID * SAM_DIM * sizeof(float));
}

/* Mask decoder forward pass. */

/* y = W x + b for a PyTorch [nout, nin] weight, optional ReLU. */
static void host_linear(const float *W, const float *b, const float *x, int nin,
                        int nout, int relu, float *y)
{
    int o, i;

    for (o = 0; o < nout; o++) {
        const float *w = W + (long)o * nin;
        double acc = b[o];

        for (i = 0; i < nin; i++)
            acc += (double)w[i] * x[i];
        y[o] = relu && acc < 0.0 ? 0.0f : (float)acc;
    }
}

static void host_mlp3(float *const W[3], float *const B[3], const float *x,
                      int nout, float *y)
{
    float h1[SAM_DIM], h2[SAM_DIM];

    host_linear(W[0], B[0], x, SAM_DIM, SAM_DIM, 1, h1);
    host_linear(W[1], B[1], h1, SAM_DIM, SAM_DIM, 1, h2);
    host_linear(W[2], B[2], h2, SAM_DIM, nout, 0, y);
}

/* Token -> image cross attention on per-prompt keys, residual into the
 * queries (layers >= 1 and the final attention). */
static void t2i_attention(struct sam_model *m, int P,
                          const struct sam_attn_w *a)
{
    const long T = SAM_GRID * SAM_GRID, n = (long)P * NTOK * SAM_DIM;

    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    add_bcast(m, m->kin, m->keys, m->pe_img, P * T * SAM_DIM, T * SAM_DIM);
    linear(m, P * NTOK, SAM_DIM, 128, m->qin, 0, a->qw, a->qb, m->sq, 0,
           ACT_NONE, 0);
    linear(m, P * T, SAM_DIM, 128, m->kin, 0, a->kw, a->kb, m->bk, 0, ACT_NONE,
           0);
    linear(m, P * T, SAM_DIM, 128, m->keys, 0, a->vw, a->vb, m->bv, 0, ACT_NONE,
           0);
    mha(m, P, NTOK, T, 128, m->sq, NTOK * 128, m->bk, T * 128, m->bv, T * 128,
        m->so);
    linear(m, P * NTOK, 128, SAM_DIM, m->so, 0, a->ow, a->ob, m->qry, 0,
           ACT_NONE, 1);
}

static void token_mlp(struct sam_model *m, int P, const struct sam_dec_layer *L)
{
    linear(m, P * NTOK, SAM_DIM, 2048, m->qry, 0, L->m1w, L->m1b, m->sh, 0,
           ACT_RELU, 0);
    linear(m, P * NTOK, 2048, SAM_DIM, m->sh, 0, L->m2w, L->m2b, m->qry, 0,
           ACT_NONE, 1);
}

void sam_decode(struct sam_model *m, int P, const float *points, float *iou)
{
    const long T = SAM_GRID * SAM_GRID, n = (long)P * NTOK * SAM_DIM;
    const struct sam_dec_layer *L0 = &m->layers[0], *L1 = &m->layers[1];
    float *tok = m->h_queries, *hyper;
    int p, i;

    if (P > m->batch)
        G_fatal_error("sam_decode: batch overflow");

    /* Prompt tokens: IoU token, 4 mask tokens, the positive point and
     * the padding point SAM appends when no box is given. */
    for (p = 0; p < P; p++) {
        float *t = tok + (long)p * NTOK * SAM_DIM;

        memcpy(t, m->iou_token, SAM_DIM * sizeof(float));
        memcpy(t + SAM_DIM, m->mask_tokens, 4 * SAM_DIM * sizeof(float));
        pe_encode(m, (points[2 * p] + 0.5f) / SAM_IMG,
                  (points[2 * p + 1] + 0.5f) / SAM_IMG, t + 5 * SAM_DIM);
        for (i = 0; i < SAM_DIM; i++) {
            t[5 * SAM_DIM + i] += m->point_emb[1][i];
            t[6 * SAM_DIM + i] = m->not_a_point[i];
        }
    }
    ocl_write(m->ocl, m->qpe, 0, tok, n * sizeof(float));
    ocl_check(clEnqueueCopyBuffer(m->ocl->queue, m->qpe, m->qry, 0, 0,
                                  n * sizeof(float), 0, NULL, NULL),
              "clEnqueueCopyBuffer");

    /* Layer 0: self attention without positional encoding and without
     * residual (skip_first_layer_pe). */
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qry, 0, L0->self.qw, L0->self.qb,
           m->sq, 0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qry, 0, L0->self.kw, L0->self.kb,
           m->sk, 0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qry, 0, L0->self.vw, L0->self.vb,
           m->sv, 0, ACT_NONE, 0);
    mha(m, P, NTOK, NTOK, SAM_DIM, m->sq, NTOK * SAM_DIM, m->sk, NTOK * SAM_DIM,
        m->sv, NTOK * SAM_DIM, m->so);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->so, 0, L0->self.ow, L0->self.ob,
           m->qry, 0, ACT_NONE, 0);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L0->n1w, L0->n1b, DEC_EPS,
              ACT_NONE);

    /* Layer 0 token -> image: keys are still the shared image embedding,
     * so the cached projections serve every prompt (stride 0). */
    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    linear(m, P * NTOK, SAM_DIM, 128, m->qin, 0, L0->t2i.qw, L0->t2i.qb, m->sq,
           0, ACT_NONE, 0);
    mha(m, P, NTOK, T, 128, m->sq, NTOK * 128, m->kp0, 0, m->vp0, 0, m->so);
    linear(m, P * NTOK, 128, SAM_DIM, m->so, 0, L0->t2i.ow, L0->t2i.ob, m->qry,
           0, ACT_NONE, 1);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L0->n2w, L0->n2b, DEC_EPS,
              ACT_NONE);
    token_mlp(m, P, L0);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L0->n3w, L0->n3b, DEC_EPS,
              ACT_NONE);

    /* Layer 0 image -> token, queries from the shared embedding. */
    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    linear(m, P * NTOK, SAM_DIM, 128, m->qin, 0, L0->i2t.kw, L0->i2t.kb, m->sk,
           0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, 128, m->qry, 0, L0->i2t.vw, L0->i2t.vb, m->sv,
           0, ACT_NONE, 0);
    mha(m, P, T, NTOK, 128, m->qp0, 0, m->sk, NTOK * 128, m->sv, NTOK * 128,
        m->bo);
    linear(m, P * T, 128, SAM_DIM, m->bo, 0, L0->i2t.ow, L0->i2t.ob, m->keys, 0,
           ACT_NONE, 0);
    add_bcast(m, m->keys, m->keys, m->src, P * T * SAM_DIM, T * SAM_DIM);
    layernorm(m, P * T, SAM_DIM, m->keys, m->keys, L0->n4w, L0->n4b, DEC_EPS,
              ACT_NONE);

    /* Layer 1. */
    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qin, 0, L1->self.qw, L1->self.qb,
           m->sq, 0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qin, 0, L1->self.kw, L1->self.kb,
           m->sk, 0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qry, 0, L1->self.vw, L1->self.vb,
           m->sv, 0, ACT_NONE, 0);
    mha(m, P, NTOK, NTOK, SAM_DIM, m->sq, NTOK * SAM_DIM, m->sk, NTOK * SAM_DIM,
        m->sv, NTOK * SAM_DIM, m->so);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->so, 0, L1->self.ow, L1->self.ob,
           m->qry, 0, ACT_NONE, 1);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L1->n1w, L1->n1b, DEC_EPS,
              ACT_NONE);
    t2i_attention(m, P, &L1->t2i);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L1->n2w, L1->n2b, DEC_EPS,
              ACT_NONE);
    token_mlp(m, P, L1);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L1->n3w, L1->n3b, DEC_EPS,
              ACT_NONE);

    /* Layer 1 image -> token (kin = keys + pe from t2i_attention). */
    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    linear(m, P * T, SAM_DIM, 128, m->kin, 0, L1->i2t.qw, L1->i2t.qb, m->bq, 0,
           ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, 128, m->qin, 0, L1->i2t.kw, L1->i2t.kb, m->sk,
           0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, 128, m->qry, 0, L1->i2t.vw, L1->i2t.vb, m->sv,
           0, ACT_NONE, 0);
    mha(m, P, T, NTOK, 128, m->bq, T * 128, m->sk, NTOK * 128, m->sv,
        NTOK * 128, m->bo);
    linear(m, P * T, 128, SAM_DIM, m->bo, 0, L1->i2t.ow, L1->i2t.ob, m->keys, 0,
           ACT_NONE, 1);
    layernorm(m, P * T, SAM_DIM, m->keys, m->keys, L1->n4w, L1->n4b, DEC_EPS,
              ACT_NONE);

    /* Final token -> image attention. */
    t2i_attention(m, P, &m->final_t2i);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, m->nfw, m->nfb, DEC_EPS,
              ACT_NONE);

    /* Output upscaling: ConvT 256->64, LayerNorm2d, GELU, ConvT 64->32,
     * GELU. The kin buffer is free again and holds the first GEMM. */
    gemm(m, P * T, 256, SAM_DIM, mat(m->keys, 0, SAM_DIM), mat(m->up0w, 0, 256),
         0, mat(m->kin, 0, 256), 1, 1, NULL, NULL, 0, 1.0f, ACT_NONE);
    convt_scatter(m, m->kin, m->u1, m->up0b, P, SAM_GRID, SAM_GRID, 64,
                  ACT_NONE);
    layernorm(m, P * 4 * T, 64, m->u1, m->u1, m->up1w, m->up1b, ENC_EPS,
              ACT_GELU);
    gemm(m, P * 4 * T, 128, 64, mat(m->u1, 0, 64), mat(m->up3w, 0, 128), 0,
         mat(m->g2, 0, 128), 1, 1, NULL, NULL, 0, 1.0f, ACT_NONE);
    convt_scatter(m, m->g2, m->u2, m->up3b, P, 2 * SAM_GRID, 2 * SAM_GRID, 32,
                  ACT_GELU);

    /* Hypernetwork MLPs and IoU head are tiny: run them on the host. */
    ocl_read(m->ocl, m->qry, 0, tok, n * sizeof(float));
    hyper = G_malloc((size_t)P * 96 * sizeof(float));
#pragma omp parallel for private(i)
    for (p = 0; p < P; p++) {
        const float *t = tok + (long)p * NTOK * SAM_DIM;
        float out[4];

        for (i = 1; i <= 3; i++)
            host_mlp3(m->hyper_w[i], m->hyper_b[i], t + (1 + i) * SAM_DIM, 32,
                      hyper + (long)p * 96 + (i - 1) * 32);
        host_mlp3(m->iou_w, m->iou_b, t, 4, out);
        for (i = 0; i < 3; i++)
            iou[p * 3 + i] = out[1 + i];
    }
    ocl_write(m->ocl, m->hyper, 0, hyper, (size_t)P * 96 * sizeof(float));
    G_free(hyper);

    {
        cl_kernel k = m->k_logits;
        cl_int npix = SAM_LOWRES * SAM_LOWRES;
        cl_int total = P * npix;

        SETARG(k, 0, m->u2);
        SETARG(k, 1, m->hyper);
        SETARG(k, 2, m->lowres);
        SETARG(k, 3, npix);
        SETARG(k, 4, total);
        run1d(m, k, total, "mask_logits");
    }
}

void sam_postprocess(struct sam_model *m, int n, const int *sel, int in_h,
                     int in_w, int out_h, int out_w, float offset, int *stats)
{
    cl_kernel k = m->k_post;
    size_t need = (size_t)n * out_h * out_w, g[3], l[3] = {16, 16, 1};
    cl_int cih = in_h, ciw = in_w, coh = out_h, cow = out_w;
    cl_float coff = offset;
    int i;

    if (n <= 0)
        return;
    if (need > m->mask_cap) {
        ocl_release(&m->mask);
        m->mask = ocl_alloc(m->ocl, need, "postprocessed masks");
        m->mask_cap = need;
    }
    for (i = 0; i < n; i++) {
        stats[6 * i + 0] = 0;
        stats[6 * i + 1] = 0;
        stats[6 * i + 2] = 0x7fffffff;
        stats[6 * i + 3] = 0x7fffffff;
        stats[6 * i + 4] = -1;
        stats[6 * i + 5] = -1;
    }
    ocl_write(m->ocl, m->sel, 0, sel, n * sizeof(cl_int));
    ocl_write(m->ocl, m->stats, 0, stats, 6 * n * sizeof(cl_int));

    SETARG(k, 0, m->lowres);
    SETARG(k, 1, m->sel);
    SETARG(k, 2, cih);
    SETARG(k, 3, ciw);
    SETARG(k, 4, coh);
    SETARG(k, 5, cow);
    SETARG(k, 6, coff);
    SETARG(k, 7, m->mask);
    SETARG(k, 8, m->stats);
    g[0] = (size_t)(out_w + 15) / 16 * 16;
    g[1] = (size_t)(out_h + 15) / 16 * 16;
    g[2] = n;
    run(m, k, 3, g, l, "mask_post");
    ocl_read(m->ocl, m->stats, 0, stats, 6 * n * sizeof(cl_int));
}

void sam_read_mask(struct sam_model *m, int k, int out_h, int out_w, int x0,
                   int y0, int x1, int y1, unsigned char *dst)
{
    size_t origin[3] = {(size_t)x0, (size_t)y0, (size_t)k};
    size_t region[3] = {(size_t)(x1 - x0 + 1), (size_t)(y1 - y0 + 1), 1};

    ocl_check(clEnqueueReadBufferRect(m->ocl->queue, m->mask, CL_TRUE, origin,
                                      (size_t[]){0, 0, 0}, region, out_w,
                                      (size_t)out_w * out_h, region[0], 0, dst,
                                      0, NULL, NULL),
              "clEnqueueReadBufferRect");
}

void sam_free(struct sam_model *m)
{
    int p;

    for (p = 0; p < nprof; p++)
        G_message("kernel %-24s %8.1f ms %7ld calls", prof[p].what, prof[p].ms,
                  prof[p].calls);

    cl_mem *bufs[] = {
        &m->pe_w,  &m->pe_b,   &m->pos,    &m->neck0,  &m->neck1w, &m->neck1b,
        &m->neck2, &m->neck3w, &m->neck3b, &m->nfw,    &m->nfb,    &m->up0w,
        &m->up0b,  &m->up1w,   &m->up1b,   &m->up3w,   &m->up3b,   &m->img,
        &m->x,     &m->xn,     &m->xw,     &m->qkv,    &m->scores, &m->attn,
        &m->tmp,   &m->mlpbuf, &m->emb,    &m->src,    &m->pe_img, &m->srcpe,
        &m->kp0,   &m->vp0,    &m->qp0,    &m->qpe,    &m->qry,    &m->qin,
        &m->sq,    &m->sk,     &m->sv,     &m->so,     &m->sh,     &m->keys,
        &m->kin,   &m->bq,     &m->bk,     &m->bv,     &m->bo,     &m->u1,
        &m->g2,    &m->u2,     &m->hyper,  &m->lowres, &m->mask,   &m->sel,
        &m->stats};
    cl_kernel kernels[] = {m->k_gemm,      m->k_gemm_big, m->k_attn_few,
                           m->k_attn_many, m->k_ln,       m->k_softmax,
                           m->k_patch,     m->k_add,      m->k_wpart,
                           m->k_wunpart,   m->k_im2col,   m->k_convt,
                           m->k_logits,    m->k_post};
    size_t i;
    int j;

    for (i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
        ocl_release(bufs[i]);
    for (j = 0; j < m->depth; j++) {
        struct sam_block *b = &m->blocks[j];

        ocl_release(&b->n1w);
        ocl_release(&b->n1b);
        ocl_release(&b->relh);
        ocl_release(&b->relw);
        ocl_release(&b->qkvw);
        ocl_release(&b->qkvb);
        ocl_release(&b->projw);
        ocl_release(&b->projb);
        ocl_release(&b->n2w);
        ocl_release(&b->n2b);
        ocl_release(&b->l1w);
        ocl_release(&b->l1b);
        ocl_release(&b->l2w);
        ocl_release(&b->l2b);
    }
    G_free(m->blocks);
    for (j = 0; j < 2; j++) {
        struct sam_dec_layer *L = &m->layers[j];
        cl_mem *lb[] = {&L->n1w, &L->n1b, &L->n2w, &L->n2b, &L->n3w, &L->n3b,
                        &L->n4w, &L->n4b, &L->m1w, &L->m1b, &L->m2w, &L->m2b};

        release_attn(&L->self);
        release_attn(&L->t2i);
        release_attn(&L->i2t);
        for (i = 0; i < sizeof(lb) / sizeof(lb[0]); i++)
            ocl_release(lb[i]);
    }
    release_attn(&m->final_t2i);
    for (j = 0; j < 4; j++) {
        for (i = 0; i < 3; i++) {
            G_free(m->hyper_w[j][i]);
            G_free(m->hyper_b[j][i]);
        }
    }
    for (i = 0; i < 3; i++) {
        G_free(m->iou_w[i]);
        G_free(m->iou_b[i]);
    }
    for (i = 0; i < sizeof(kernels) / sizeof(kernels[0]); i++)
        if (kernels[i])
            clReleaseKernel(kernels[i]);
    G_free(m->h_queries);
    memset(m, 0, sizeof(*m));
}
