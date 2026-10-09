#!/usr/bin/env python3
"""Read existing run logs and verify scalar algebra; never starts training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def collect_runs(code_root):
    base = code_root / 'reproduction/Preconditioned_Inexact_Stochastic_ADMM_for_Deep_Models/runs/fig2/nano'
    runs = [('adam', 'full'), ('muon', 'full-v2'), ('nsisa', 'full-v1'),
            ('soap', 'full-v2'), ('pbsgdm(优化参数)', 'full'),
            ('lp-sgda(优化参数)', 'full'), ('lp-sgdma(优化参数)', 'full')]
    rows = []
    inputs = []
    for name, version in runs:
        folder = base / name / version
        file = folder / 'run_summary.json'
        data = json.loads(file.read_text())
        keys = ['val_loss_last', 'wall_clock_hours', 'tokens_total',
                'tokens_per_s', 'peak_memory_MiB']
        row = {key: data[key] for key in keys}
        row['run'] = name + '/' + version
        inputs.append(file)
        if name.startswith('lp-'):
            file = folder / 'train_metrics.jsonl'
            events = [json.loads(line) for line in file.read_text().splitlines()]
            rates = [event['train/learning_rate'] for event in events
                     if event.get('event') == 'train_step' and 'train/learning_rate' in event]
            row.update(n=len(rates), cap_fraction=sum(rate > .99 / 300 for rate in rates) / len(rates),
                       eta_median=statistics.median(rates), eta_final=rates[-1])
            inputs.append(file)
        rows.append(row)
    manifest = [{'path': str(file.relative_to(code_root)),
                 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()} for file in inputs]
    return rows, manifest


def check_mechanisms():
    rows = []
    for gradient in [1., .1, .01, .001, .00001]:
        c = 300.
        eta = gradient / (gradient ** 2 + c * gradient + 1e-8)
        direction = gradient ** .3
        power_eta = gradient * direction / (gradient ** 2 + c * gradient * direction + 1e-8)
        rows.append(dict(gradient=gradient, aps_muon_eta=eta, c_eta=c * eta,
                         step_norm_if_unit_muon=eta,
                         lp_gamma_03_step_norm=direction * power_eta))
    tests = []
    for loss_scale in [.1, 1., 10.]:
        for direction_scale in [.2, 1., 5.]:
            k = 2. * loss_scale
            x = .01
            direction = direction_scale
            delta = .001
            probe_coefficient = delta / abs(direction)
            gradient = k * x
            probe_gradient = k * (x - probe_coefficient * direction)
            alignment = gradient * direction
            curvature = (gradient * direction - probe_gradient * direction) / probe_coefficient
            alpha = alignment / curvature
            displacement = alpha * direction
            if not math.isclose(displacement, x, rel_tol=1e-12):
                raise AssertionError((loss_scale, direction_scale, displacement))
            tests.append(dict(loss_scale=loss_scale, direction_scale=direction_scale,
                              alpha=alpha, displacement=displacement))
    return dict(scope='Deterministic scalar algebra checks only; not a training experiment.',
                aps_degree_zero=rows, quadratic_secant_invariance=tests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path,
                        default=Path.home() / 'Documents/graduate-thesis-code')
    parser.add_argument('--output-dir', type=Path, help='Optional NEW directory; refuses to overwrite.')
    args = parser.parse_args()
    rows, manifest = collect_runs(args.code_root)
    checks = check_mechanisms()
    outputs = dict(run_evidence=rows, mechanism_checks=checks, run_input_manifest=manifest)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in outputs:
            if (args.output_dir / (name + '.json')).exists():
                raise FileExistsError('Refusing to overwrite ' + name)
        for name, data in outputs.items():
            (args.output_dir / (name + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(runs=rows, quadratic_scaling_checks=len(checks['quadratic_secant_invariance'])),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
