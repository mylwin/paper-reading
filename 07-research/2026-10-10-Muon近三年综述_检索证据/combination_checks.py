"""Deterministic algebra only; no training and no third-party dependencies."""
import json
import math
from pathlib import Path


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def main():
    q = [[1/math.sqrt(2), 1/math.sqrt(2), 0],
         [-1/math.sqrt(6), 1/math.sqrt(6), 2/math.sqrt(6)],
         [1/math.sqrt(3), -1/math.sqrt(3), 1/math.sqrt(3)]]
    powered = [[math.copysign(math.sqrt(abs(x)), x) if x else 0 for x in row] for row in q]
    original_dot = dot(q[1], q[2])
    powered_dot = dot(powered[1], powered[2])
    assert abs(original_dot) < 1e-12 and abs(powered_dot) > .1
    # Spectral powers followed by exact polar preserve singular vectors and support.
    spectral = []
    for alpha in [.1, .5, 1]:
        values = [s ** alpha for s in [9, 1, .01]]
        polar_values = [x/abs(x) for x in values]
        assert polar_values == [1, 1, 1]
        spectral.append(dict(alpha=alpha, input_singular_values=values,
                             exact_polar_singular_values=polar_values))
    # Same update ray, APS does not transform its coefficient inversely.
    aps = []
    g, d, c = .2, 1., 3.
    for k in [.2, 1, 5]:
        ak = abs(g*k*d)
        eta = ak/(g*g+c*ak)
        aps.append(dict(direction_scale=k, eta=eta, displacement=eta*k*d))
    assert max(x['displacement'] for x in aps)-min(x['displacement'] for x in aps) > .1
    # A one-dimensional quadratic curvature probe identifies the optimum radius.
    secant = []
    for scale in [.1, 1, 10]:
        curvature, x, delta = 2*scale, .01, .001
        a = curvature*x
        probe = curvature*(x-delta)
        estimated_curvature = (a-probe)/delta
        radius = a/estimated_curvature
        assert math.isclose(radius, x, rel_tol=1e-12)
        secant.append(dict(loss_scale=scale, curvature=estimated_curvature, radius=radius))
    result = dict(scope='Algebraic sanity checks; not optimizer convergence or LLM evidence.',
                  coordinate_power_breaks_orthogonality=dict(original_row_dot=original_dot,
                                                           powered_row_dot=powered_dot),
                  spectral_power_then_exact_polar=spectral,
                  aps_direction_rescaling=aps,
                  quadratic_radius_loss_scale_invariance=secant)
    path = Path(__file__).with_name('combination_checks.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
