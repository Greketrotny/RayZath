#ifndef CUDA_INCLUDE_H
#define CUDA_INCLUDE_H

#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "device_atomic_functions.h"
#include "texture_fetch_functions.h"
#include "surface_functions.h"


#ifdef __CUDACC__
#define cui_sinf __sinf
#define cui_cosf __cosf
#define cui_sincosf __sincosf
#define cui_powf __powf
#define cui_logf __logf
#else
#define cui_sinf sinf
#define cui_cosf cosf
#define cui_sincosf(a, s, c)
#define cui_powf powf
#define cui_logf logf
#endif // __CUDACC__


#endif // !CUDA_INCLUDE_H