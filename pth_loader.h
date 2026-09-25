#ifndef I_SAM_OPENCL_PTH_LOADER_H
#define I_SAM_OPENCL_PTH_LOADER_H

#include <stdio.h>
#include <sys/types.h>

#define PTH_MAX_DIMS 8

enum pth_dtype { PTH_F32, PTH_F16, PTH_BF16 };

/* One tensor of a PyTorch state_dict, described by where its storage
 * lives inside the (uncompressed) zip archive and how it is strided
 * over that storage. */
struct pth_tensor {
    char *name;
    int ndim;
    long shape[PTH_MAX_DIMS];
    long stride[PTH_MAX_DIMS];
    long storage_offset; /* In elements. */
    enum pth_dtype dtype;
    off_t data_offset;    /* Byte offset of the storage in the file. */
    size_t storage_bytes; /* Byte size of the storage entry. */
};

struct pth_file {
    FILE *fp;
    char *path;
    int ntensors;
    struct pth_tensor *tensors;
};

/* Open a PyTorch checkpoint (torch.save zip format, stored entries) and
 * index its state_dict. Calls G_fatal_error() on any format problem. */
void pth_open(struct pth_file *pf, const char *path);

/* Return the tensor called name, or NULL if absent. */
const struct pth_tensor *pth_find(const struct pth_file *pf, const char *name);

/* Like pth_find(), but fatal if the tensor is missing or its shape does
 * not match the ndim leading entries of shape[] (-1 matches anything). */
const struct pth_tensor *pth_require(const struct pth_file *pf,
                                     const char *name, int ndim,
                                     const long *shape);

long pth_numel(const struct pth_tensor *t);

/* Read a tensor as a newly allocated, C-contiguous float32 array. */
float *pth_read_f32(const struct pth_file *pf, const struct pth_tensor *t);

void pth_close(struct pth_file *pf);

#endif /* I_SAM_OPENCL_PTH_LOADER_H */
