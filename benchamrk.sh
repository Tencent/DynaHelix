# Copyright 2026 Tencent and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

sh exp.sh casp base_constraint iface
sleep 10

sh exp.sh fb_AA base_constraint iface
sleep 10

sh exp.sh fb_PP base_constraint iface
sleep 10

sh exp.sh fb_dsDNA dyna_55199 iface