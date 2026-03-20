# Copyright 2026 Weitang Ye
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

./bin/build_artea_graph \
    --beta 1.44 \
    --ul-max-nbr-size 64 \
    --bl-max-nbr-size 64 \
    --ul-scale-coeffs 1.10 \
    --bl-scale-coeffs 1.10 \
    --ul-shifted-coeffs 0.10 \
    --bl-shifted-coeffs 0.10 \
    --dataset sift-1m