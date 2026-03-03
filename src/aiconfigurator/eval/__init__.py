# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

warnings.warn(
    "The 'aiconfigurator.eval' module is deprecated and will be removed in a future release. "
    "Please migrate to the standalone benchmarking and deployment tooling.",
    DeprecationWarning,
    stacklevel=2,
)
