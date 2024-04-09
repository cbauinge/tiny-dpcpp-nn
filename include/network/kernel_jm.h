/**
 * @file kernel_jm.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief SYCL joint matrix (jm) implementation of the forward, backward and inference kernels.
 * @version 0.1
 * @date 2024-04-08
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <sycl/sycl.hpp>
#include <vector>

#include "DeviceMatrix.h"
#include "common.h"
#include "kernel_common.h"
#include "oneapi/mkl.hpp"
#include "sycl/ext/oneapi/matrix/matrix-unified.hpp"
#include "sycl/ext/oneapi/matrix/matrix.hpp"

namespace tinydpcppnn {
namespace kernels {
namespace jm {

using namespace sycl::ext::oneapi::experimental::matrix;
using namespace sycl;
using sycl::ext::intel::experimental::esimd::cache_hint;

/**
 * @brief class which wraps the joint matrix kernels
 *
 * @tparam T type for the computations. Everything that is supported by xmx::dpas
 * should work fine.
 * @tparam INPUT_WIDTH The width of the input layer of the network. In general
 * it should be a multiple of TK, right now it is equal to WIDTH.
 * @tparam WIDTH Denotes the width of every hidden layer, may be 16, 32, 64, 128.
 * @tparam OUTPUT_WIDTH The width of the output layer, currently equal to WIDTH. Later a multiple of TN.
 * @tparam activation Activation function. Currently either none or ReLU.
 * @tparam output_activation Activation for the output layer. Currently None.
 * @tparam TN Device dependent, whatever is supported by the chosen device. 8 for DG2, 16 for PVC.
 */
template <typename T, int INPUT_WIDTH, int WIDTH, int OUTPUT_WIDTH, Activation activation, Activation output_activation>
class JMKernels {

    using Tc = typename XMXCType<T>::CType;
    static constexpr int TN = XMXTn::TN;

  public:
    static std::vector<sycl::event> forward_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                 const DeviceMatrixView<T> &input,
                                                 DeviceMatricesView<T> intermediate_output, const int n_hidden_layers,
                                                 const std::vector<sycl::event> &deps) {
        return forward_impl_general<false>(q, weights, input, intermediate_output, n_hidden_layers, deps);
    }

    static std::vector<sycl::event> backward_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                  const DeviceMatrixView<T> &input, DeviceMatricesView<T> output,
                                                  DeviceMatricesView<T> intermediate_backward,
                                                  const DeviceMatricesView<T> &intermediate_forward,
                                                  const int n_hidden_layers, const std::vector<sycl::event> &deps,
                                                  std::optional<DeviceMatrixView<T>> dL_dinput = std::nullopt) {
        throw std::logic_error("Backward pass not implemented in SYCL version.");
    }

    static std::vector<sycl::event> inference_impl(sycl::queue &q, const DeviceMatricesView<T> &weights,
                                                   const DeviceMatrixView<T> &input,
                                                   DeviceMatricesView<T> intermediate_output, const int n_hidden_layers,
                                                   const std::vector<sycl::event> &deps) {
        return forward_impl_general<true>(q, weights, input, intermediate_output, n_hidden_layers, deps);
    }

    /*************the following functions are only public for testing purposes*******************/

    // in register everything is in block major format with blocks of size TMxTK
    template <int TM, int TK, cache_hint L1, cache_hint L3, typename Group, use Use, layout Layout, size_t Nmats>
    static void storeRow(Group &sg, const std::array<joint_matrix<Group, T, Use, TM, TK, Layout>, Nmats> &src,
                         T *const dest) {

        static_assert(TM == 1 || TM == 2 || TM == 4 || TM == 8);
        static_assert(WIDTH % TK == 0);
        static_assert(sizeof(T) <= 4);

#pragma unroll
        for (int iter = 0; iter < Nmats; iter++) {
            sycl::ext::intel::experimental::matrix::joint_matrix_store(sg, src[iter],
                                                                       sycl::global_ptr<T>(dest + iter * TK), WIDTH);
        }
    }

    // in register everything is in block major format with blocks of size TMxTK
    template <int TM, int TK, cache_hint L1, cache_hint L3, typename Group, use Use, layout Layout, size_t Nmats>
    static void loadRow(Group &sg, T const *const src,
                        std::array<joint_matrix<Group, T, Use, TM, TK, Layout>, Nmats> &dest) {
        static_assert(TM == 1 || TM == 2 || TM == 4 || TM == 8);
        static_assert(WIDTH % TK == 0);
        static_assert(sizeof(T) <= 4);

#pragma unroll
        for (int iter = 0; iter < Nmats; iter++) {
            joint_matrix_load(sg, dest[iter], sycl::global_ptr<const T>(src + iter * TK), WIDTH);
        }
    }

    // we are assuming a block major layout and vnni'd B
    template <int TM, int TK, size_t Nmats, typename Group, layout LayoutA>
    static void MAD(Group &sg, const std::array<joint_matrix<Group, T, use::a, TM, TK, LayoutA>, Nmats> &As,
                    T const *const __restrict__ B,
                    std::array<joint_matrix<Group, Tc, use::accumulator, TM, TN>, Nmats> &Cs) {
        static_assert(TM >= 1 && TM <= 8);
        static_assert(TN == 16 || TN == 8);
        static_assert(WIDTH % TK == 0 && WIDTH % TN == 0);
        static_assert(sizeof(T) <= 4 && sizeof(Tc) <= 4);

        constexpr int vnni_factor = std::max<int>(1, 4 / sizeof(T));

#pragma collapse(2) unroll
        for (int iterA = 0; iterA < Nmats; iterA++) {
            for (int iterC = 0; iterC < Nmats; iterC++) {
                joint_matrix<Group, T, use::b, TK, TN, layout::ext_intel_packed> mB;
                joint_matrix_load(sg, mB, sycl::global_ptr<const T>(B + iterC * TN * vnni_factor + iterA * TK * WIDTH),
                                  WIDTH * vnni_factor);
                joint_matrix_mad(sg, Cs[iterC], As[iterA], mB, Cs[iterC]);
            }
        }
    }

    template <Activation act, int TM, int TK, typename Group, typename Tsrc, typename Tdest, use UseSrc, use UseDest,
              layout LayoutSrc, layout LayoutDest, size_t Nmats>
    static void applyActivation(Group &sg, std::array<joint_matrix<Group, Tsrc, UseSrc, TM, TK, LayoutSrc>, Nmats> &Src,
                                std::array<joint_matrix<Group, Tdest, UseDest, TM, TN, LayoutDest>, Nmats> &Dest) {
        static_assert(TM >= 1 && TM <= 8);
        static_assert(TN == 16 || TN == 8);
        static_assert(TK == 8 || TK == 16 || TK == 32 || TK == 64);

        for (int iter = 0; iter < Nmats; iter++) {
            auto data_Src = sycl::ext::oneapi::detail::get_wi_data(sg, Src[iter]);
            auto data_Dest = sycl::ext::oneapi::detail::get_wi_data(sg, Dest[iter]);
            // if constexpr (act == Activation::None) {
            joint_matrix_copy(sg, Src[iter], Dest[iter]);
            //} else if constexpr (act == Activation::ReLU) {
            //    joint_matrix_apply(sg, Src[iter], [=](Tsrc &x) { x = std::max(static_cast<Tsrc>(0), x); });
            //    joint_matrix_copy(sg, Src[iter], Dest[iter]);
            //}
        }
    }

  private:
    template <bool INFERENCE>
    static std::vector<sycl::event>
    forward_impl_general(sycl::queue &q, const DeviceMatricesView<T> &weights, const DeviceMatrixView<T> &input,
                         DeviceMatricesView<T> intermediate_output, const int n_hidden_layers,
                         const std::vector<sycl::event> &deps) {

        // throw std::logic_error("General function should not be called.");
        const size_t M = input.m();
        static_assert(INPUT_WIDTH == WIDTH);
        static_assert(OUTPUT_WIDTH == WIDTH);
        static_assert(WIDTH % TN == 0);

        constexpr int TM = ComputeTM<WIDTH>();
        // make sure there is no remainder and no out of bounds accesses
        // this may be adjusted in the future
        assert(M % TM == 0);

        // TK depends on the datatype T
        constexpr int TK = ComputeTK<T>();
        static_assert(TN == TK);
        const int ITEMS_IN_WG = ComputeSGsInWG(M, TM) * TN;

        // One Block Row has TM rows an N columns.
        auto e = q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(deps);

            cgh.parallel_for(
                sycl::nd_range<1>(M / TM * TN, ITEMS_IN_WG),
                [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(TN)]] {
                    auto sg = item.get_sub_group();
                    const size_t loc_row_offset =
                        (sg.get_group_linear_range() * item.get_group_linear_id() + sg.get_group_linear_id()) * TM;

                    // we store blocks contiguously
                    std::array<joint_matrix<sycl::sub_group, T, use::a, TM, TK, layout::row_major>, WIDTH / TK> As;
                    loadRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(sg, input.GetPointer(loc_row_offset, 0),
                                                                                As);

                    // if not inference activate and store in intermediate output
                    if constexpr (!INFERENCE) {
                        storeRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(
                            sg, As,
                            intermediate_output.GetElementPointer(0, loc_row_offset, 0)); // saving non-activated input
                    }

                    std::array<joint_matrix<sycl::sub_group, Tc, use::accumulator, TM, TN>, WIDTH / TN> Cs;
                    for (int layer = 0; layer < n_hidden_layers; layer++) {
                        // reset result matrices
                        for (auto &C : Cs) {
                            joint_matrix_fill(sg, C, static_cast<Tc>(0));
                        }

                        MAD<TM, TK>(sg, As, weights.GetMatrixPointer(layer), Cs);

                        // activate and save
                        applyActivation<activation, TM, TK>(sg, Cs, As);

                        if constexpr (!INFERENCE)
                            storeRow<TM, TK, cache_hint::uncached, cache_hint::uncached>(
                                sg, As, intermediate_output.GetElementPointer(layer + 1, loc_row_offset, 0));
                    }

                    // generate output, i.e. last GEMM
                    for (auto &C : Cs) {
                        joint_matrix_fill(sg, C, static_cast<Tc>(0));
                    }

                    MAD<TM, TK>(sg, As, weights.GetMatrixPointer(n_hidden_layers), Cs);

                    // activate
                    applyActivation<output_activation, TM, TK>(sg, Cs, As);

                    // save to HBM
                    if constexpr (!INFERENCE)
                        storeRow<TM, TK, cache_hint::uncached, cache_hint::write_back>(
                            sg, As, intermediate_output.GetElementPointer(n_hidden_layers + 1, loc_row_offset, 0));
                    else if constexpr (INFERENCE) // storing at the beginning since no intermediate results
                        storeRow<TM, TK, cache_hint::uncached, cache_hint::write_back>(
                            sg, As, intermediate_output.GetElementPointer(0, loc_row_offset, 0));
                });
        });

        return {e};
    }
};

} // namespace jm
} // namespace kernels
} // namespace tinydpcppnn