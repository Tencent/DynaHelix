// modified from https://github.com/NVIDIA/apex/blob/master/csrc/compat.h

#ifndef PROTENIX_OPENFOLD_LOCAL_UTILS_KERNEL_CSRC_COMPAT_H_
#define PROTENIX_OPENFOLD_LOCAL_UTILS_KERNEL_CSRC_COMPAT_H_

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#ifdef VERSION_GE_1_3
#define DATA_PTR data_ptr
#else
#define DATA_PTR data
#endif

#endif  // PROTENIX_OPENFOLD_LOCAL_UTILS_KERNEL_CSRC_COMPAT_H_
