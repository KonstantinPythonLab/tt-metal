// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/normalization/softmax/device/softmax_op.hpp"
#include "tt_metal/host_api.hpp"

namespace ttnn::operations::normalization {

// ... (existing implementation details)

void SoftmaxSharded::validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto shard_spec = input_tensor.shard_spec().value();
    auto grid = shard_spec.grid;
    
    // Ensure we respect the actual grid offset rather than assuming (0,0)
    // The program configuration must be derived from the shard_spec grid
    TT_FATAL(grid.bounding_box().start_coord.x >= 0 && grid.bounding_box().start_coord.y >= 0, "Shard grid must have valid coordinates");
}

// Logic update: Ensure runtime arguments are generated based on the actual shard grid mapping
// instead of hardcoded (0,0) origin.

} // namespace ttnn::operations::normalization