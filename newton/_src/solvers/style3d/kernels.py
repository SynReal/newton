# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...geometry import ParticleFlags


@wp.func
def triangle_deformation_gradient(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, inv_dm: wp.mat22):
    x01, x02 = x1 - x0, x2 - x0
    Fu = x01 * inv_dm[0, 0] + x02 * inv_dm[1, 0]
    Fv = x01 * inv_dm[0, 1] + x02 * inv_dm[1, 1]
    return Fu, Fv


@wp.kernel
def eval_stretch_kernel(
    pos: wp.array[wp.vec3],
    face_areas: wp.array[float],
    inv_dms: wp.array[wp.mat22],
    faces: wp.array2d[wp.int32],
    aniso_ke: wp.array[wp.vec3],
    # outputs
    forces: wp.array[wp.vec3],
):
    """
    Ref. Large Steps in Cloth Simulation, Baraff & Witkin in 1998.
    """
    fid = wp.tid()

    inv_dm = inv_dms[fid]
    face_area = face_areas[fid]
    face = wp.vec3i(faces[fid, 0], faces[fid, 1], faces[fid, 2])

    Fu, Fv = triangle_deformation_gradient(pos[face[0]], pos[face[1]], pos[face[2]], inv_dm)

    len_Fu = wp.length(Fu)
    len_Fv = wp.length(Fv)

    Fu = wp.normalize(Fu) if (len_Fu > 1e-6) else wp.vec3(0.0)
    Fv = wp.normalize(Fv) if (len_Fv > 1e-6) else wp.vec3(0.0)

    dFu_dx = wp.vec3(-inv_dm[0, 0] - inv_dm[1, 0], inv_dm[0, 0], inv_dm[1, 0])
    dFv_dx = wp.vec3(-inv_dm[0, 1] - inv_dm[1, 1], inv_dm[0, 1], inv_dm[1, 1])

    ku = aniso_ke[fid][0]
    kv = aniso_ke[fid][1]
    ks = aniso_ke[fid][2]

    for i in range(3):
        force = -face_area * (
            ku * (len_Fu - 1.0) * dFu_dx[i] * Fu
            + kv * (len_Fv - 1.0) * dFv_dx[i] * Fv
            + ks * wp.dot(Fu, Fv) * (Fu * dFv_dx[i] + Fv * dFu_dx[i])
        )
        wp.atomic_add(forces, face[i], force)


@wp.kernel
def eval_bend_kernel(
    pos: wp.array[wp.vec3],
    edge_rest_area: wp.array[float],
    edge_bending_cot: wp.array[wp.vec4],
    edges: wp.array2d[wp.int32],
    edge_bending_properties: wp.array2d[float],
    # outputs
    forces: wp.array[wp.vec3],
):
    """
    Crouzeix-Raviart isometric bending model from

    "A Quadratic Bending Model for Inextensible Surfaces" (Bergou et al. 2006).

    For one interior edge with local stencil x = (x0, x1, x2, x3)^T,
    the paper defines

        E_b = 1/2 * k * x^T Q x,
        Q = 3 / (A0 + A1) * w^T w,

    where A0 and A1 are the incident triangle rest areas and w is built
    from rest-pose cotangents. The conservative force is

        F_i = -dE_b/dx_i = -k * sum_j Q_ij x_j.
    """
    eid = wp.tid()
    if edges[eid][0] < 0 or edges[eid][1] < 0:
        return
    edge = edges[eid]
    edge_stiff = edge_bending_properties[eid][0] * (3.0 / edge_rest_area[eid])
    bend_weight = wp.vec4(0.0)
    bend_weight[2] = edge_bending_cot[eid][2] + edge_bending_cot[eid][3]
    bend_weight[3] = edge_bending_cot[eid][0] + edge_bending_cot[eid][1]
    bend_weight[0] = -edge_bending_cot[eid][0] - edge_bending_cot[eid][2]
    bend_weight[1] = -edge_bending_cot[eid][1] - edge_bending_cot[eid][3]
    for i in range(4):
        force = wp.vec3(0.0)
        for j in range(4):
            force = force - edge_stiff * bend_weight[i] * bend_weight[j] * pos[edge[j]]
        wp.atomic_add(forces, edge[i], force)


@wp.kernel
def eval_drag_force_kernel(
    spring_stiff: float,
    face_index: wp.array[int],
    drag_pos: wp.array[wp.vec3],
    drag_bary_coord: wp.array[wp.vec3],
    faces: wp.array2d[wp.int32],
    vert_pos: wp.array[wp.vec3],
    # outputs
    forces: wp.array[wp.vec3],
):
    fid = face_index[0]
    if fid != -1:
        coord = drag_bary_coord[0]
        face = wp.vec3i(faces[fid, 0], faces[fid, 1], faces[fid, 2])
        x0 = vert_pos[face[0]]
        x1 = vert_pos[face[1]]
        x2 = vert_pos[face[2]]
        p = x0 * coord[0] + x1 * coord[1] + x2 * coord[2]
        dir = drag_pos[0] - p

        # add force
        force = spring_stiff * dir
        wp.atomic_add(forces, face[0], force * coord[0])
        wp.atomic_add(forces, face[1], force * coord[1])
        wp.atomic_add(forces, face[2], force * coord[2])

        # add hessian
        # dir = wp.normalize(dir)
        # hessian = wp.outer(dir, dir) * spring_stiff
        # hessian_diags[face[0]] += hessian * coord[0]
        # hessian_diags[face[1]] += hessian * coord[1]
        # hessian_diags[face[2]] += hessian * coord[2]


@wp.kernel
def accumulate_dragging_pd_diag_kernel(
    spring_stiff: float,
    face_index: wp.array[int],
    drag_bary_coord: wp.array[wp.vec3],
    faces: wp.array2d[wp.int32],
    particle_flags: wp.array[wp.int32],
    # outputs
    pd_diags: wp.array[float],
):
    fid = face_index[0]
    if fid != -1:
        coord = drag_bary_coord[0]
        face = wp.vec3i(faces[fid, 0], faces[fid, 1], faces[fid, 2])

        if particle_flags[face[0]] & ParticleFlags.ACTIVE:
            pd_diags[face[0]] += spring_stiff * coord[0]

        if particle_flags[face[1]] & ParticleFlags.ACTIVE:
            pd_diags[face[1]] += spring_stiff * coord[1]

        if particle_flags[face[2]] & ParticleFlags.ACTIVE:
            pd_diags[face[2]] += spring_stiff * coord[2]


@wp.kernel
def init_step_kernel(
    dt: float,
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    f_ext: wp.array[wp.vec3],
    v_curr: wp.array[wp.vec3],
    x_curr: wp.array[wp.vec3],
    x_prev: wp.array[wp.vec3],
    pd_diags: wp.array[float],
    particle_masses: wp.array[float],
    particle_flags: wp.array[wp.int32],
    # outputs
    x_inertia: wp.array[wp.vec3],
    static_A_diags: wp.array[float],
    dx: wp.array[wp.vec3],
):
    tid = wp.tid()
    x_last = x_curr[tid]
    x_prev[tid] = x_last

    if not particle_flags[tid] & ParticleFlags.ACTIVE:
        x_inertia[tid] = x_prev[tid]
        static_A_diags[tid] = 0.0
        dx[tid] = wp.vec3(0.0)
    else:
        v_prev = v_curr[tid]
        mass = particle_masses[tid]
        static_A_diags[tid] = pd_diags[tid] + mass / (dt * dt)
        world_idx = particle_world[tid]
        world_g = gravity[wp.max(world_idx, 0)]
        x_inertia[tid] = x_last + v_prev * dt + (world_g + f_ext[tid] / mass) * (dt * dt)
        dx[tid] = v_prev * dt

        # temp
        # x_curr[tid] = x_last + v_prev * dt


@wp.kernel
def init_inertia_warm_start_kernel(
    x_inertia: wp.array[wp.vec3],
    x_curr: wp.array[wp.vec3],
    # outputs
    dx: wp.array[wp.vec3],
):
    """ITER1: seed the first nonlinear iterate's PCG guess with the FULL inertial step.

    ``init_step_kernel`` leaves ``dx = v_prev * dt``; the inertial target is
    ``x_inertia = x + v*dt + (g + f/m)*dt^2``, so the stock guess is short by
    exactly the gravity/external-force displacement of the substep.  On the
    global-translation mode there is no stiffness to drive that residual, so a
    truncated PCG never recovers it.  This writes ``dx = x_inertia - x_curr``
    instead, i.e. it hands PCG the free-flight step and lets it solve only for
    the elastic correction.  Inactive particles get 0 either way, because
    ``init_step_kernel`` sets ``x_inertia = x_prev = x_curr`` for them.
    """
    tid = wp.tid()
    dx[tid] = x_inertia[tid] - x_curr[tid]


# ------------------------------------------------------------------ ITER2
# The ITER1 guess above moves EVERY particle by the full free-flight step
# (g*dt^2 = 39 um at dt = 2 ms).  A particle that is in contact should not
# move: the contact stack reads positions inside the Newton loop (the tri-SDF
# tangential anchor measures slip against the moved point, the cloth-cloth
# penalties are evaluated there), and on the non-TP path iteration 0 gets a
# single PCG step, which cannot pull the offset back out.  The kernels below
# mark the particles that carry a contact pair THIS substep, and the gated
# guess leaves those on the stock `dx = v_prev*dt` while every free particle
# keeps the full ITER1 warm start.
#
# The marks are evaluated on x_prev -- `Collision.frame_begin` has already run
# and `init_step_kernel` has not touched positions -- so they describe THIS
# substep with no lag.  What is marked:
#
#   * cloth-cloth VF and EE: the broad-phase candidate lists, filtered by the
#     SAME narrow-phase predicate the force kernels use.  The raw candidate
#     list cannot be used as the gate: it is a proximity list (max_dist =
#     3*radius = 9 mm with a 3 mm AABB pad), so on a 5-7 mm mesh every vertex
#     has 2-ring neighbours in it and the gate marks 100 % of the cloth
#     (measured: 4674/4674 on silk_4k).  With the predicate applied it marks
#     only pairs that actually produce a force.
#   * tri-SDF: the T14 gather's centroid-vs-blade-AABB test, re-run here with
#     the same `half_thickness + reach + slack`.  That one IS kept broad (a
#     pair a few mm out at iteration 0 can be driven into the shell by the
#     solve, which is the same per-substep contact-set rule the tri-SDF force
#     kernel runs under), and it does not depend on T14_SDF_BROADPHASE.
#   * NOT marked: the EF untangling channel (its broad phase has the same
#     density problem and its predicate only fires on an edge that already
#     crosses a face -- those vertices are inside the VF/EE thickness anyway),
#     and the stock particle-vs-shape channel (table/links).  The latter is
#     deliberate: the ITER1 audit measured 0.000 mm of warm-start effect on
#     `gsilk`, a single layer lying flat on the table, so that channel is not
#     the one carrying the regression.
#
# Writes are idempotent stores of 1, so the concurrent marks race benignly and
# the result does not depend on thread order.


@wp.func
def iter2_triangle_normal(A: wp.vec3, B: wp.vec3, C: wp.vec3):
    """Copy of ``collision.kernels.triangle_normal`` (keeps ITER2 to two files)."""
    n = wp.cross(B - A, C - A)
    ln = wp.length(n)
    return wp.vec3(0.0) if ln < 1.0e-12 else (n / ln)


@wp.func
def iter2_triangle_barycentric(A: wp.vec3, B: wp.vec3, C: wp.vec3, P: wp.vec3):
    """Copy of ``collision.kernels.triangle_barycentric``."""
    v0 = A - C
    v1 = B - C
    v2 = P - C
    dot00 = wp.dot(v0, v0)
    dot01 = wp.dot(v0, v1)
    dot02 = wp.dot(v0, v2)
    dot11 = wp.dot(v1, v1)
    dot12 = wp.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    invDenom = 0.0 if wp.abs(denom) < 1.0e-12 else 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return wp.vec3(u, v, 1.0 - u - v)


@wp.kernel
def iter2_gate_mark_vf_kernel(
    thickness: float,
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    broad_phase_vf: wp.array2d[int],
    # outputs
    gate: wp.array[wp.int32],
):
    """Mark a VF pair that passes ``handle_vertex_triangle_contacts_kernel``'s test."""
    vid = wp.tid()
    count = wp.min(broad_phase_vf[0, vid], 31)
    if count <= 0:
        return
    x0 = pos[vid]
    for i in range(count):
        fid = broad_phase_vf[i + 1, vid]
        f0 = tri_indices[fid, 0]
        f1 = tri_indices[fid, 1]
        f2 = tri_indices[fid, 2]
        x1 = pos[f0]
        x2 = pos[f1]
        x3 = pos[f2]
        tri_normal = iter2_triangle_normal(x1, x2, x3)
        dist = wp.dot(x0 - x1, tri_normal)
        if wp.abs(dist) > thickness:
            continue
        bary = iter2_triangle_barycentric(x1, x2, x3, x0 - tri_normal * dist)
        if bary[0] < 0.0 or bary[1] < 0.0 or bary[2] < 0.0:
            continue
        gate[vid] = 1
        gate[f0] = 1
        gate[f1] = 1
        gate[f2] = 1


@wp.kernel
def iter2_gate_mark_ee_kernel(
    thickness: float,
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    broad_phase_ee: wp.array2d[int],
    # outputs
    gate: wp.array[wp.int32],
):
    """Mark an EE pair that passes ``handle_edge_edge_contacts_kernel``'s test.

    Columns 2/3 are the edge's own endpoints (0/1 are the opposite vertices of
    the two adjacent triangles) -- same convention as the force kernel, whose
    adjacency-limited thickness is reproduced here.
    """
    eid = wp.tid()
    count = wp.min(broad_phase_ee[0, eid], 31)
    if count <= 0:
        return
    edge0 = wp.vec4i(edge_indices[eid, 2], edge_indices[eid, 3], edge_indices[eid, 0], edge_indices[eid, 1])
    x0 = pos[edge0[0]]
    x1 = pos[edge0[1]]
    len0 = wp.length(x0 - x1)
    for i in range(count):
        idx = broad_phase_ee[i + 1, eid]
        edge1 = wp.vec4i(
            edge_indices[idx, 2], edge_indices[idx, 3], edge_indices[idx, 0], edge_indices[idx, 1]
        )
        x2 = pos[edge1[0]]
        x3 = pos[edge1[1]]
        st = wp.closest_point_edge_edge(x0, x1, x2, x3, wp.float32(1e-5))
        s = st[0]
        t = st[1]
        if (s <= 0.0) or (s >= 1.0) or (t <= 0.0) or (t >= 1.0):
            continue
        dist = wp.length(wp.lerp(x0, x1, s) - wp.lerp(x2, x3, t))
        limited_thickness = thickness
        avg_len = (len0 + wp.length(x2 - x3)) * 0.5
        if edge0[2] == edge1[0] or edge0[3] == edge1[0]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        elif edge0[2] == edge1[1] or edge0[3] == edge1[1]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        if edge1[2] == edge0[0] or edge1[3] == edge0[0]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        elif edge1[2] == edge0[1] or edge1[3] == edge0[1]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        if 1.0e-6 < dist and dist < limited_thickness:
            gate[edge0[0]] = 1
            gate[edge0[1]] = 1
            gate[edge1[0]] = 1
            gate[edge1[1]] = 1


@wp.func
def iter2_point_aabb_dist2(p: wp.vec3, lo: wp.vec3, hi: wp.vec3):
    """Squared distance from ``p`` to the axis-aligned box; 0 when inside.

    Same expression as ``collision.kernels._point_aabb_dist2``.
    """
    dx = wp.max(wp.max(lo[0] - p[0], p[0] - hi[0]), 0.0)
    dy = wp.max(wp.max(lo[1] - p[1], p[1] - hi[1]), 0.0)
    dz = wp.max(wp.max(lo[2] - p[2], p[2] - hi[2]), 0.0)
    return dx * dx + dy * dy + dz * dz


@wp.kernel
def iter2_gate_mark_tri_sdf_kernel(
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    tri_count: int,
    slot_shape: wp.array[int],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    aabb_lo: wp.array[wp.vec3],
    aabb_hi: wp.array[wp.vec3],
    half_thickness: float,
    slack: float,
    # outputs
    gate: wp.array[wp.int32],
):
    """Mark the 3 vertices of every triangle that can reach a tri-SDF blade.

    One thread per (slot, triangle) pair; the test is ``tri_sdf_broadphase_kernel``'s,
    so the marked set is a superset of the pairs the contact kernel can
    evaluate this substep.
    """
    tid = wp.tid()
    slot = tid / tri_count
    t = tid - slot * tri_count

    shape = slot_shape[slot]
    body = shape_body[shape]
    X_ws = shape_transform[shape]
    if body >= 0:
        X_ws = body_q[body] * shape_transform[shape]
    X_sw = wp.transform_inverse(X_ws)

    i0 = tri_indices[t, 0]
    i1 = tri_indices[t, 1]
    i2 = tri_indices[t, 2]
    a = wp.transform_point(X_sw, pos[i0])
    b = wp.transform_point(X_sw, pos[i1])
    c = wp.transform_point(X_sw, pos[i2])

    third = 1.0 / 3.0
    g = (a + b + c) * third
    reach = wp.max(wp.length(a - g), wp.max(wp.length(b - g), wp.length(c - g)))
    r = half_thickness + reach + slack

    if iter2_point_aabb_dist2(g, aabb_lo[slot], aabb_hi[slot]) <= r * r:
        gate[i0] = 1
        gate[i1] = 1
        gate[i2] = 1


@wp.kernel
def iter2_gate_count_kernel(
    gate: wp.array[wp.int32],
    # outputs
    stat: wp.array[wp.int32],
):
    """Count the marked particles (self-provenance readback, rule 35).

    ``stat[0]`` is zeroed by the caller before every substep.
    """
    tid = wp.tid()
    if gate[tid] != 0:
        wp.atomic_add(stat, 0, 1)


@wp.kernel
def iter2_gate_peak_kernel(
    stat: wp.array[wp.int32],
    # outputs
    peak: wp.array[wp.int32],
):
    """Carry the per-run maximum of the substep count, device side.

    Without it the provenance line would report whatever the LAST substep of
    the run happened to see, which is 0 for a task that ends with the cloth
    lying free -- indistinguishable from a dead gate.
    """
    wp.atomic_max(peak, 0, stat[0])


@wp.kernel
def init_inertia_warm_start_free_kernel(
    x_inertia: wp.array[wp.vec3],
    x_curr: wp.array[wp.vec3],
    gate: wp.array[wp.int32],
    # outputs
    dx: wp.array[wp.vec3],
):
    """ITER2: the ITER1 guess, applied only to particles with no contact pair.

    A gated particle is left with whatever ``init_step_kernel`` wrote
    (``v_prev*dt`` when active, 0 when not), which is exactly the stock guess,
    so ``free`` degrades to the OFF path wherever the cloth is in contact and
    to the ``True`` path wherever it is flying.
    """
    tid = wp.tid()
    if gate[tid] == 0:
        dx[tid] = x_inertia[tid] - x_curr[tid]


# ------------------------------------------------------------------ ITER2b
@wp.kernel
def init_accel_warm_start_kernel(
    dt: float,
    v_prev: wp.array[wp.vec3],
    v_prev2: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    # outputs
    dx: wp.array[wp.vec3],
):
    """ITER2b: constant-acceleration predictor for the first nonlinear iterate.

    The guess is only a starting point for Newton; the requirement on it is
    "close to the true solution", and the true displacement of a substep is
    ``v*dt + a*dt^2`` with ``a`` including the CONTACT force.  ITER1 used
    ``a = g``, which is exact in free flight but pushes a particle resting on a
    support ``g*dt^2 = 39 um`` into it every substep.  Here ``a`` is measured
    from the two previous substeps instead:

        a_prev = (v_prev - v_prev2) / dt
        dx     = v_prev*dt + a_prev*dt^2

    In free flight ``a_prev = g`` (exact, same as ITER1); at rest on a support
    ``a_prev = 0`` (exact, same as the stock guess).  It is only wrong on the
    single substep where the acceleration jumps, and it self-corrects on the
    next one, so nothing accumulates.  No particle classification, no contact
    lookup, no discontinuity between neighbours, no parameter.

    Inactive particles are skipped so they keep the ``dx = 0`` that
    ``init_step_kernel`` wrote.
    """
    tid = wp.tid()
    if not particle_flags[tid] & ParticleFlags.ACTIVE:
        return
    a_prev = (v_prev[tid] - v_prev2[tid]) / dt
    dx[tid] = v_prev[tid] * dt + a_prev * dt * dt


# ------------------------------------------------------------------ ITER2c
@wp.kernel
def iter2c_contact_force_delta_kernel(
    rhs: wp.array[wp.vec3],
    # in/out: on entry the snapshot of `rhs` taken BEFORE the contact pass,
    # on exit the contact contribution alone
    f_contact: wp.array[wp.vec3],
):
    """Isolate the contact force out of the shared right-hand side.

    ``rhs`` carries inertia + stretch + bend + (drag) before
    ``Collision.accumulate_contact_force`` adds the contact terms into the same
    array, so the contact part is exactly the difference across that call.
    """
    tid = wp.tid()
    f_contact[tid] = rhs[tid] - f_contact[tid]


@wp.kernel
def init_accel_f_warm_start_kernel(
    dt: float,
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    v_prev: wp.array[wp.vec3],
    f_contact: wp.array[wp.vec3],
    particle_masses: wp.array[float],
    particle_flags: wp.array[wp.int32],
    # outputs
    dx: wp.array[wp.vec3],
):
    """ITER2c: first-iterate guess from gravity + the PREVIOUS substep's contact force.

        dx = v_prev*dt + (g + f_c_prev/m)*dt^2

    ``f_c_prev`` is the contact force accumulated at the LAST nonlinear
    iteration of the previous substep -- contact only, no elastic/bending term.

    Why this is not the ITER2b predictor: nothing here is extrapolated from a
    velocity difference, so there is no ``v_{n+1} = 2 v_n - v_{n-1}`` recursion
    and no ``(z-1)^2`` double root.  In free flight ``f_c_prev = 0`` and the
    guess is identically ITER1's (exact); at rest on a support ``f_c_prev =
    +m g`` and the guess is identically the stock ``v_prev*dt`` (no push into
    the support).  The only feedback path left runs through the contact force
    itself, and only on particles that actually have contact.

    First substep: ``f_contact`` is still zero, i.e. the guess degenerates to
    ITER1's.  Inactive particles keep the ``dx = 0`` that ``init_step_kernel``
    wrote.
    """
    tid = wp.tid()
    if not particle_flags[tid] & ParticleFlags.ACTIVE:
        return
    world_g = gravity[wp.max(particle_world[tid], 0)]
    mass = particle_masses[tid]
    accel = world_g
    if mass > 0.0:
        accel = world_g + f_contact[tid] / mass
    dx[tid] = v_prev[tid] * dt + accel * dt * dt


@wp.kernel
def init_rhs_kernel(
    dt: float,
    x_curr: wp.array[wp.vec3],
    x_inertia: wp.array[wp.vec3],
    particle_masses: wp.array[float],
    # outputs
    rhs: wp.array[wp.vec3],
):
    tid = wp.tid()
    rhs[tid] = (x_inertia[tid] - x_curr[tid]) * particle_masses[tid] / (dt * dt)


@wp.kernel
def prepare_jacobi_preconditioner_kernel(
    static_A_diags: wp.array[float],
    contact_hessian_diags: wp.array[wp.mat33],
    particle_flags: wp.array[wp.int32],
    # outputs
    inv_A_diags: wp.array[wp.mat33],
):
    tid = wp.tid()
    diag = wp.identity(3, float) * static_A_diags[tid]
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        diag += contact_hessian_diags[tid]
    inv_A_diags[tid] = wp.inverse(diag) if static_A_diags[tid] > 0.0 else wp.identity(3, float) * 0.0


@wp.kernel
def prepare_jacobi_preconditioner_no_contact_hessian_kernel(
    static_A_diags: wp.array[float],
    # outputs
    inv_A_diags: wp.array[wp.mat33],
):
    tid = wp.tid()
    diag = wp.identity(3, float) * static_A_diags[tid]
    inv_A_diags[tid] = wp.inverse(diag) if static_A_diags[tid] > 0.0 else wp.identity(3, float) * 0.0


@wp.kernel
def PD_jacobi_step_kernel(
    rhs: wp.array[wp.vec3],
    x_in: wp.array[wp.vec3],
    inv_diags: wp.array[wp.mat33],
    # outputs
    x_out: wp.array[wp.vec3],
):
    tid = wp.tid()
    x_out[tid] = x_in[tid] + inv_diags[tid] * rhs[tid]


@wp.kernel
def nonlinear_step_kernel(
    x_in: wp.array[wp.vec3],
    # outputs
    x_out: wp.array[wp.vec3],
    dx: wp.array[wp.vec3],
):
    tid = wp.tid()
    x_out[tid] = x_in[tid] + dx[tid]
    dx[tid] = wp.vec3(0.0)


@wp.kernel
def eval_aero_force_kernel(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    faces: wp.array2d[wp.int32],
    tri_materials: wp.array2d[float],
    # outputs
    f_aero: wp.array[wp.vec3],
):
    """AERO1: per-triangle aerodynamic drag + lift, accumulated per particle.

    Same force law as newton's semi-implicit solver
    (``solvers/semi_implicit/kernels_particle.py``), read from the same two
    ``tri_materials`` columns (3 = drag, 4 = lift)::

        f_drag = k_drag * A * |n . v_mid| * v_mid
        f_lift = k_lift * A * (pi/2 - acos(n . v_hat)) * |v_mid|^2 * n

    Both act AGAINST the triangle's motion; the total is split evenly over the
    three vertices.  ``k_drag = 0.5 * rho * C_d`` in SI, so for air at
    ``rho = 1.2`` and a flat plate ``C_d ~ 1.2`` the physical value is ~0.7.

    Skipped triangle-wise when both coefficients are zero, so a model built
    with the stock defaults (0.0) contributes exactly nothing.
    """
    fid = wp.tid()

    k_drag = tri_materials[fid, 3]
    k_lift = tri_materials[fid, 4]
    if k_drag == 0.0 and k_lift == 0.0:
        return

    i = faces[fid, 0]
    j = faces[fid, 1]
    k = faces[fid, 2]

    x0 = pos[i]
    cr = wp.cross(pos[j] - x0, pos[k] - x0)
    area2 = wp.length(cr)
    if area2 <= 0.0:
        return
    n = cr / area2
    area = 0.5 * area2

    v_mid = (vel[i] + vel[j] + vel[k]) / 3.0
    v_len = wp.length(v_mid)
    if v_len <= 0.0:
        return
    v_dir = v_mid / v_len

    f_drag = v_mid * (k_drag * area * wp.abs(wp.dot(n, v_mid)))
    f_lift = n * (k_lift * area * (wp.HALF_PI - wp.acos(wp.clamp(wp.dot(n, v_dir), -1.0, 1.0))) * v_len * v_len)

    f_vert = -(f_drag + f_lift) / 3.0
    wp.atomic_add(f_aero, i, f_vert)
    wp.atomic_add(f_aero, j, f_vert)
    wp.atomic_add(f_aero, k, f_vert)


@wp.kernel
def update_velocity(
    dt: float,
    vel_damping: float,
    prev_pos: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
):
    particle = wp.tid()
    vel[particle] = vel_damping * (pos[particle] - prev_pos[particle]) / dt
