import numpy as np


def dgp_area_yield(
    seed=None,
    n_obs=5000,
    K=100,
    # origin
    origin_shape='ellipsis',
    origin_a=0.035,
    origin_b=0.01,
    origin_pertubation=0.1,
    # target
    target_center=(1.5, 0),
    target_a=0.6,
    target_b=0.3,
    target_shape='ellipsis',
    # action
    action_shift=(1.0, 0),
    action_scale=1.02,
    action_pertubation=(0.001, 0.0006),
    action_drag_share=0.7,
    action_drag_scale=0.5,
    # running variables
    running_dist_measure='projected',
    running_mea_selection=5,
    # treatment
    treatment_dist=0.45,
    treatment_improvement=0,
    treatment_random_share=0.001
):
    """
    The dgp mimicks a productions process where the yield of a production lot
    consisting of `K` individual items should be optimized.
    Each item is described by a 2D location at the plane.
    The initial datapoints for a lot are randomly sampled from a region specified by
    - `origin_shape` (either equal or gaussian distribution)
    - `origin_a`, `origin_b` determining the range of the shape in x and y direction.

    The goal is to move all the items of a lot into a target region.
    This region is specified through `target_center`, `target_a`, `target_b` and `target_shape`
    and have the same meaning as for the origin.

    Some production lots are quite good to begin with and
    hit the target region very well while others are not.
    This is modeled through a random placement of the center of
    the data points between the coordinate origin and the `target_center`.
    Additionally `origin_pertubation` describes
    the magnitude of random shifts applied to the points in orthogonal
    direction of the target vector.

    To improve the situation for a lot a action can be issued that shifts
    the whole point cloud of the lot along the action vector `action_shift`.
    But be aware that applying the action also induces some random pertubation effects on the points:
        - `action_scale`: expansion of the point cloud fixing it's center
        - `action_pertubation`: magnitude of random pertubations applied to the `action_shift` vector
        - `action_drag_share` and `action_drag_scale`: some randomly selected points get dragged behind.

    The running variable `X` consists of two decision criterias:
    1. the mean distance from the target `X1`
    2. yield improvement `X2` estimated by the decision maker
    Note that the the decision maker estimates the yield improvment performing
    a hypothetical `action_shift` without pertubations on some selected
    lot items he measured (because of cost consiterations not all items in a lot might be measured).
    This is controlled by `running_mea_selection` (e.g. measure every k-th item).
    `running_dist_measure` controls the used distance measure for the first decision criteria.
    As the `action_shift` is fixed the `euclidean` distance
    does also include the orthogonal displacement of the target, where as `projected` does not.

    Regarding the treatment the folloing parameters are relevant:
        - `treatment_dist`: defines the cutoff for the distance criteria `X1`
        - `treatment_improvement`: defines the cuttof for the estimated yield improvement `X2`
        - `treatment_random_share`: some decisionmakers might defy this

    Note that defiers can also be caused by the partial information the decisonmaker has!
    """
    rnd = np.random.default_rng(seed)

    state = _generate_initial_states(
        rnd, n_obs, K,
        target_center, origin_a, origin_b, origin_shape, origin_pertubation,
    )
    measured_state = state[:, ::running_mea_selection, :]
    estimated_state = measured_state + np.array(action_shift)
    treated_state = _execute_action(
        rnd,
        n_obs, K,
        state,
        action_shift,
        action_pertubation,
        action_scale,
        action_drag_share,
        action_drag_scale
    )

    # estimated yield + yield
    y0_est, _ = _check_yield(measured_state, target_center, target_a, target_a, target_shape)
    y1_est, _ = _check_yield(estimated_state, target_center, target_a, target_a, target_shape)
    y0, _ = _check_yield(state, target_center, target_a, target_b, target_shape)
    y1, _ = _check_yield(treated_state, target_center, target_a, target_b, target_shape)

    # running variables
    center_est = np.mean(measured_state, axis=1)
    center = np.mean(state, axis=1)

    if running_dist_measure == 'projected':
        # magnitude in action_shift direction: <center - target, e_0> e_0
        e_0 = action_shift / np.linalg.norm(action_shift)
        distance_est = np.matmul(target_center - center_est, e_0)
        distance = np.matmul(target_center - center, e_0)

    elif running_dist_measure == 'euclidean':
        distance_est = np.linalg.norm(center_est - target_center, axis=1)
        distance = np.linalg.norm(center - target_center, axis=1)
    else:
        raise ValueError('unkown distance measure')

    improvement_est = y1_est - y0_est
    improvement = y1 - y0

    # treatment decision
    if treatment_dist is None:
        treatment_dist = np.linalg.norm(action_shift)
    assinged_treatment = (distance_est >= treatment_dist) & (improvement_est > treatment_improvement)

    # we assume that the decision maker knows the state better
    actual_treatment = (distance >= treatment_dist) & (improvement > treatment_improvement)
    if treatment_random_share > 0:
        n_rnd = int(n_obs*treatment_random_share)
        actual_treatment[:n_rnd] = rnd.choice([True, False], size=n_rnd)

    # select observed entries
    y_obs = np.where(actual_treatment, y1, y0)
    state_obs = np.where(
        np.expand_dims(np.expand_dims(actual_treatment, 1), 2),
        treated_state, state
    )

    return {
        'state': state,
        'treated_state': treated_state,
        'final_state': state_obs,
        'Z': measured_state,
        'Y0': y0,
        'Y1': y1,
        'Y': y_obs,
        'X1': distance_est,
        'X2': improvement_est,
        'X1_act': distance,
        'X2_act': improvement,
        'T': assinged_treatment,
        'D': actual_treatment
    }


def _execute_action(
    rnd,
    n_obs,
    K,
    state,
    action_shift,
    action_pertubation,
    action_scale,
    action_drag_share,
    action_drag_scale,
):
    treated_state = state + np.array(action_shift)

    # action pertubation
    if action_pertubation is not None:
        # pertubation = rnd.multivariate_normal(
        #     [0, 0],
        #     [[action_pertubation[0], 0],
        #      [0, action_pertubation[1]]],
        #     n_obs*K
        # )
        # pertubation = np.expand_dims(state, 0).reshape(n_obs, K, 2)
        # treated_state = pertubation + treated_state

        # systematic pertubation of whole obs (n_obs, 1, 2)
        pertubation = rnd.multivariate_normal(
            [0, 0],
            [[action_pertubation[0], 0],
             [0, action_pertubation[1]]],
            n_obs
        )
        pertubation = np.expand_dims(pertubation, 1)
        treated_state = pertubation + treated_state

    # action scale up
    if action_scale > 1:
        center = np.expand_dims(np.mean(treated_state, axis=1), 1)
        lambda_s = np.expand_dims(np.expand_dims(rnd.uniform(1, action_scale, n_obs), 1), 1)
        treated_state = center + (treated_state - center) * lambda_s

    # drag behind
    if action_drag_share > 0.0:
        # choose points to drag (n_obs, K)
        drag_mask = rnd.choice(
            [0, 1], size=(n_obs, K),
            replace=True, p=[1-action_drag_share, action_drag_share]
        )
        # choose drag lambda (n_obs, K, 2)
        drag_force = np.random.uniform(0, action_drag_scale, n_obs*K)
        drag_force = np.expand_dims(drag_force, 0).reshape(n_obs, K)
        drag_force = drag_mask * drag_force
        drag_force = np.repeat(np.expand_dims(drag_force, 2), 2, axis=2)
        # drag
        treated_state = treated_state - drag_force * np.array(action_shift)

    return treated_state


def _generate_initial_states(
    rnd,
    n_obs,
    K,
    target_center,
    origin_a,
    origin_b,
    origin_shape,
    origin_pertubation,
):
    """Initial states centered around (0,0); shape (n_obs, K, 2)."""
    if origin_shape == 'ellipsis':
        state = rnd.multivariate_normal(
            [0, 0],
            [[origin_a, 0], [0, origin_b]],
            n_obs*K
        )
        state = np.expand_dims(state, 0).reshape(-1, K, 2)
    elif origin_shape == 'rectangle':
        state = (
            rnd.uniform(-origin_a/2, origin_a/2, n_obs*K),
            rnd.uniform(-origin_b/2, origin_b/2, n_obs*K)
        )
        state = np.expand_dims(np.stack(state, 1), 0).reshape(-1, K, 2)
    else:
        raise ValueError('invalid origin_shape')

    # TODO pertubate single entries

    # randomly shift original shape towards target (n_obs, 1)
    lambda_t = np.expand_dims(rnd.uniform(0, 1.2, n_obs), 1)
    # shift (n_obs, 2)
    shift = lambda_t * np.array([target_center]).repeat(n_obs, axis=0)
    # expand to state shape (n_obs, K, 2)
    shift = np.repeat(np.expand_dims(shift, 1), K, axis=1)

    # randomly shift orthogonal
    lambda_t = np.expand_dims(rnd.normal(0, 1, n_obs), 1)
    shift_perp = np.array([-target_center[1], target_center[0]])
    shift_perp = shift_perp / np.linalg.norm(shift_perp) * origin_pertubation
    shift_perp = lambda_t * np.expand_dims(shift_perp, 0).repeat(n_obs, 0)
    shift_perp = np.expand_dims(shift_perp, 1)

    return state + shift + shift_perp


def _check_yield(state, target_center, target_a, target_b, target_shape):
    """Calc yield out of ellipsis state."""
    centered_state = state - np.array([[target_center]])
    x = centered_state[:, :, 0]
    y = centered_state[:, :, 1]

    if target_shape == 'ellipsis':
        in_spec = (x/target_a)**2 + (y/target_b)**2 <= 1
    elif target_shape == 'rectangle':
        in_spec = (x <= target_a/2) & (x >= -target_a/2)
        in_spec = in_spec & (y <= target_b/2) & (y >= -target_b/2)
    else:
        raise ValueError('invalid target_shape')

    return in_spec.mean(axis=1), in_spec
