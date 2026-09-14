# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""T44b:三角形级位置投影「不踢速度」的两个小核(开关 T44B_TRI_PROJ_NOKICK,默认关)。

为什么单独一个文件:放进 kernels.py 会改动那个模块的内容,warp 缓存要整片冷编译
(T42 实测约 26 分钟)。放在这里,既有 kernel 的源码与常量一个字节不动,只有这两个核新编译。

机理(T44 核实):Style3D 在一步的迭代循环里把 `_tri_sdf_sweep` 的投影位移直接写进
`state_out.particle_q`,循环结束后 `update_velocity` 算 `vel = 0.998 * (x_out - x_prev) / dt`,
投影位移因此整段进了速度;下一步 `x_inertia = x + dt * vel + ...` 再把这一「踢」带进惯性预测。
开关打开时:每次 sweep 把「本次 sweep 实际搬动的位移」累加进 `step_disp`,
`frame_end` 里从速度扣掉 `0.998 * step_disp / dt` 并清零 ⇒ 投影只改位置、不进速度。
"""

import warp as wp


@wp.kernel
def t44b_accumulate_sweep_displacement_kernel(
    pre_q: wp.array[wp.vec3],
    post_q: wp.array[wp.vec3],
    step_disp: wp.array[wp.vec3],
):
    i = wp.tid()
    step_disp[i] = step_disp[i] + (post_q[i] - pre_q[i])


@wp.kernel
def t44b_remove_projection_velocity_kernel(
    scale: float,
    step_disp: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
):
    # scale = 0.998 / dt,与 solver_style3d 的 update_velocity 同一个阻尼系数
    i = wp.tid()
    vel[i] = vel[i] - scale * step_disp[i]
    step_disp[i] = wp.vec3(0.0)
