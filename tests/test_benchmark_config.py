"""Run settings edited in TOML reach the training entrypoints."""

import sys
from types import SimpleNamespace

import pytest

from experiments.kuramoto.experiment.config import DEFAULT_CONFIG, load_runs
from experiments.kuramoto.scripts import run_comparison
from experiments.sphere_latent_nsde.config import (
    DEFAULT_CONFIG as SPHERE_CONFIG,
)
from experiments.sphere_latent_nsde.config import (
    load_runs as load_sphere_runs,
)
from experiments.sphere_latent_nsde.config import (
    sweep_config_at,
)


def test_default_configs_keep_method_settings_explicit():
    assert DEFAULT_CONFIG.name == SPHERE_CONFIG.name == "config.toml"
    runs = load_runs(DEFAULT_CONFIG)
    assert len(runs) == 24
    assert {run.seed for run in runs} == {0, 1, 2}
    for run in runs:
        expected = (
            10
            if run.adaptive
            else {
                "cg2": 75,
                "cfees25": 50,
                "geometric_euler_heun": 150,
                "srkmk_general_shark": 75,
            }[run.solver]
        )
        assert run.n_steps == expected
    sphere_runs = load_sphere_runs(SPHERE_CONFIG)
    assert len(sphere_runs) == 7
    assert {run.solver for run in runs} == {run.solver for run in sphere_runs}
    for run in sphere_runs:
        assert run.n_steps == run.solve_n_steps
        assert run.effective_nfe_budget == 30
        assert "checkpoint" not in run.method


def test_custom_toml_reaches_comparison_without_cli_overrides(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("""
[base]
epochs = 2
batch_size = 3
N = 4
[sweep]
seed = [7, 11]
[integrators.cfees25]
adjoints = ["reversible"]
stepping = ["adaptive"]
n_steps = 9
[integrators.cfees25.adaptive]
atol = 0.02
dtmin = 0.0002
""")
    observed = []

    def run_cell(config_path, index, config, output_dir):
        observed.append((index, config))
        return {"exit_code": 0}

    monkeypatch.setattr(run_comparison, "_run_cell", run_cell)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_comparison",
            "--config",
            str(path),
            "--output-base",
            str(tmp_path / "output"),
        ],
    )
    assert run_comparison.main() == 0
    assert [index for index, _ in observed] == [0, 1]
    assert [config.seed for _, config in observed] == [7, 11]
    for _, config in observed:
        assert (config.epochs, config.batch_size, config.N, config.n_steps) == (
            2,
            3,
            4,
            9,
        )
        assert config.adaptive and config.adaptive_atol == 0.02
        assert config.adaptive_dtmin == 0.0002


def test_sphere_integrator_settings_reach_model_and_saved_config(tmp_path):
    import jax

    from experiments.sphere_latent_nsde.config import load_config, to_toml
    from experiments.sphere_latent_nsde.factories import make_model

    path = tmp_path / "config.toml"
    path.write_text("""
[base]
h_dim = 4
z_dim = 3
[sweep]
seed = [7]
[integrators.geometric_euler_heun]
adjoints = ["direct"]
n_steps = 12
[integrators.cfees25]
adjoints = ["reversible"]
stepping = ["fixed", "adaptive"]
n_steps = 6
[integrators.cfees25.adaptive]
n_steps = 4
atol = 0.0003
dtmin = 0.0002
""")
    fixed, cf_fixed, adaptive = [sweep_config_at(path, i) for i in range(3)]
    assert fixed.seed == adaptive.seed == 7
    assert not fixed.adaptive and adaptive.adaptive
    assert fixed.solve_n_steps == 12
    assert cf_fixed.solve_n_steps == 6
    assert adaptive.solve_n_steps == 4
    assert adaptive.adaptive_atol == pytest.approx(0.0003)
    assert adaptive.adaptive_dtmin == pytest.approx(0.0002)
    for config in (fixed, cf_fixed, adaptive):
        model = make_model(config, jax.random.key(0))
        assert model.solve_num_timepoints == config.n_steps + 1
        saved = tmp_path / "saved.toml"
        saved.write_text(to_toml(config))
        assert load_config(saved) == config


@pytest.mark.parametrize("loader", [load_runs, load_sphere_runs])
@pytest.mark.parametrize(
    "settings,match",
    [
        ('adjoints = ["typo"]', "adjoints"),
        ('adjoints = ["full"]\nstepping = ["typo"]', "stepping"),
        ('adjoints = ["full"]\nn_steps = 0', "n_steps"),
    ],
)
def test_invalid_integrator_choices_fail_before_training(
    tmp_path, loader, settings, match
):
    path = tmp_path / "config.toml"
    path.write_text(f"[integrators.geometric_euler_heun]\n{settings}\n")
    with pytest.raises(ValueError, match=match):
        loader(path)


@pytest.mark.parametrize(
    "platform,stats,expected",
    [
        (
            "gpu",
            {"peak_bytes_in_use": 4096, "bytes_in_use": 512, "bytes_reserved": 8192},
            4096,
        ),
        ("cpu", {"peak_bytes_in_use": 4096}, None),
        ("gpu", None, None),
        ("gpu", {"bytes_in_use": 512}, None),
    ],
)
def test_gpu_memory_reports_peak_allocation_or_unavailable(
    monkeypatch, platform, stats, expected
):
    from experiments import metrics

    device = SimpleNamespace(platform=platform, memory_stats=lambda: stats)
    monkeypatch.setattr(metrics.jax, "devices", lambda: [device])
    assert metrics.peak_gpu_memory_bytes() == expected


@pytest.mark.parametrize("exit_code,run_count", [(0, 7), (7, 1)])
def test_sphere_sweep_uses_fresh_processes_and_propagates_failure(
    tmp_path, monkeypatch, exit_code, run_count
):
    from experiments.sphere_latent_nsde import train_activity

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(train_activity.subprocess, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_activity", str(SPHERE_CONFIG), "--output-base", str(tmp_path)],
    )
    assert train_activity.main() == exit_code
    assert len(calls) == run_count
    for index, command in enumerate(calls):
        assert command[:4] == [
            sys.executable,
            "-u",
            "-m",
            "experiments.sphere_latent_nsde.train_activity",
        ]
        assert command[command.index("--index") + 1] == str(index)
        assert command[command.index("--output-base") + 1] == str(tmp_path)
