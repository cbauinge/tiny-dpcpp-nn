/**
 * @file kernel_common.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief common definitions for all kernels
 * @version 0.1
 * @date 2024-04-08
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

using bf16 = sycl::ext::oneapi::bfloat16;

namespace tinydpcppnn {
namespace kernels {
/**
 * @brief Struct to decide type for accumulation (CType) for XMX at compile time,
 * depending on the given type T.
 *
 * Currently returns CType == T, except when T == bf16, then CType == float.
 *
 * @tparam T
 */
template <typename T> struct XMXCType {
    typedef T CType;
};
template <> struct XMXCType<bf16> {
    typedef float CType;
};
template <> struct XMXCType<sycl::half> {
#if TARGET_DEVICE == 0
    typedef sycl::half CType;
#elif TARGET_DEVICE == 1
    typedef float CType;
#endif
};

/**
 * @brief Struct which gives us the value to use for TN in the dpas instruction
 * Depending on the device.
 *
 */
struct XMXTn {
#if TARGET_DEVICE == 0
    static constexpr int TN = 16;
#elif TARGET_DEVICE == 1
    static constexpr int TN = 8;
#endif
};

/**
 * @brief Struct to give us the maximum number of bytes in a send instruction,
 * depending on the device
 *
 */
struct XMXMaxSendBytes {
#if TARGET_DEVICE == 0
    static constexpr int MaxBytes = 512;
#elif TARGET_DEVICE == 1
    static constexpr int MaxBytes = 256;
#endif
};

template <int WIDTH> static constexpr int ComputeTM() {
#if TARGET_DEVICE == 0
    return 8;
#elif TARGET_DEVICE == 1
    if constexpr (WIDTH < 64)
        return 8;
    else if constexpr (WIDTH >= 64) {
        constexpr int factor = std::max(1, WIDTH / 64); // shut up div by 0 warning
        return std::max<int>(1, 4 / factor);
    }
#endif
}

template <typename T> static constexpr int ComputeTK() { return 8 * std::min<int>(8, 32 / (8 * sizeof(T))); }

static int ComputeSGsInWG(const size_t M, const int TM) {
// TODO: 64 depends on the device. It is different for non-PVC hardware
#if TARGET_DEVICE == 0
    constexpr int max_items_per_wg = 64;
#elif TARGET_DEVICE == 1
    constexpr int max_items_per_wg = 1;
#endif
    int items_in_wg = std::min<int>(M / TM, max_items_per_wg);
    while (M / TM % items_in_wg != 0) {
        items_in_wg--;
    }
    if (items_in_wg <= 0) throw std::logic_error("Number of SGS per WG cannot be less than 1");

    return items_in_wg;
}

} // namespace kernels
} // namespace tinydpcppnn