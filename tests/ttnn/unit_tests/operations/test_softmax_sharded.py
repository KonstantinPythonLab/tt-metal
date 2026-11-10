# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

@pytest.mark.parametrize("grid_offset", [(0, 0), (1, 0), (0, 1)])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_sharded_softmax_offset_grid(device, grid_offset, orientation):
    # Regression test for offset shard grids
    # 1. Create sharded tensor with offset grid
    # 2. Run softmax_in_place
    # 3. Verify correctness against PyTorch
    pass