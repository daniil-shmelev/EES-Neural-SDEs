"""Exercise launch isolation and subprocess failures without training models."""

import contextlib
import importlib.machinery
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("experiment_launcher", str(REPO / "run"))
spec = importlib.util.spec_from_loader(loader.name, loader)
launcher = importlib.util.module_from_spec(spec)
loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for _, _, folder, config, _ in launcher.EXPERIMENTS.values():
            if config:
                path = self.root / folder / config
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# configuration fixture\n")
        for target, value in [
            ("ROOT", self.root),
            ("shutil.which", lambda _: "/bin/uv"),
        ]:
            patcher = (
                patch.object(launcher, target, value)
                if "." not in target
                else patch("shutil.which", value)
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def invoke(self, args, status=0):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(output),
            patch.object(
                launcher.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=status),
            ) as run,
        ):
            result = launcher.main(args)
        return result, run.call_args_list, output.getvalue()

    def test_default_runs_get_distinct_directories(self):
        paths = []
        for _ in range(2):
            status, calls, _ = self.invoke(["kuramoto", "--index", "0"])
            self.assertEqual(status, 0)
            cmd = calls[0].args[0]
            self.assertIn("jax-base", cmd)
            self.assertEqual(cmd[-2:], ["--index", "0"])
            path = Path(cmd[cmd.index("--output-base") + 1])
            self.assertTrue(path.is_dir())
            paths.append(path)
        self.assertNotEqual(*paths)

    def test_all_dry_runs_are_side_effect_free(self):
        before = set(self.root.rglob("*"))
        for name in launcher.EXPERIMENTS:
            with self.subTest(name=name):
                status, calls, output = self.invoke([name, "--dry-run"])
                self.assertEqual(status, 0)
                self.assertEqual(calls, [])
                self.assertIn("uv run --locked", output)
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_relocated_experiments_resolve_modules_and_outputs(self):
        for arguments, module, folder in [
            (
                ["rough-vol"],
                "neurips.stochastic_volatility.experiment.train",
                "neurips/stochastic_volatility",
            ),
            (
                ["rough-vol", "--table"],
                "neurips.stochastic_volatility.scripts.paper_table",
                None,
            ),
            (
                ["convergence-fbm"],
                "neurips.convergence_fbm.convergence",
                "neurips/convergence_fbm",
            ),
            (
                ["convergence-fbm", "--so3"],
                "experiments.convergence_fbm.scripts.so3_reversibility_verify",
                "experiments/convergence_fbm",
            ),
            (
                ["torus-memory"],
                "neurips.torus_memory.benchmark",
                "neurips/torus_memory",
            ),
        ]:
            with self.subTest(arguments=arguments):
                _, calls, _ = self.invoke(arguments)
                command = calls[0].args[0]
                self.assertEqual(command[command.index("-m") + 1], module)
                if folder:
                    self.assertTrue(
                        any(
                            str(self.root / folder / "results") in arg
                            for arg in command
                        )
                    )

    def test_adaptive_selection_comes_from_toml(self):
        config = self.root / "experiments/kuramoto/config.toml"
        config.write_text("""
[base]
epochs = 3
[sweep]
seed = [7]
[integrators.cg2]
adjoints = ["full"]
n_steps = 12
[integrators.cfees25]
adjoints = ["reversible"]
stepping = ["adaptive"]
n_steps = 25
[integrators.cfees25.adaptive]
dtmin = 2e-4
""")
        with patch.object(Path, "is_dir", return_value=True):
            _, calls, _ = self.invoke(["kuramoto"])
            command = calls[0].args[0]
            self.assertIn("adaptive-reversible", calls[0].kwargs["env"]["PYTHONPATH"])
            for flag in ("--seed", "--adjoint", "--n-steps", "--adaptive-dtmin"):
                self.assertNotIn(flag, command)
        with patch.dict(launcher.os.environ, {}, clear=True):
            _, calls, _ = self.invoke(["kuramoto", "--index", "0"])
            self.assertNotIn("PYTHONPATH", calls[0].kwargs["env"])

    def test_run_parameters_cannot_override_toml(self):
        for args in (["kuramoto", "--seeds", "0"], ["sphere", "--epochs", "1"]):
            with self.subTest(args=args), self.assertRaises(SystemExit) as raised:
                self.invoke(args)
            self.assertEqual(raised.exception.code, 2)

    def test_failure_stops_before_plot_and_propagates(self):
        status, calls, output = self.invoke(["ou"], status=7)
        self.assertEqual(status, 7)
        self.assertEqual(len(calls), 1)
        self.assertIn("Experiment failed (exit 7)", output)

    def test_explicit_existing_directory_is_preserved(self):
        marker = self.root / "keep.txt"
        marker.write_text("result")
        with self.assertRaises(SystemExit) as raised:
            self.invoke(["kuramoto", "--output-dir", str(self.root)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(marker.read_text(), "result")

    def test_help_creates_no_run_and_only_calls_help(self):
        before = set(self.root.rglob("*"))
        status, calls, _ = self.invoke(["kuramoto", "--help"])
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("--help", calls[0].args[0])
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_torch_comparisons_and_selected_method(self):
        for name, count in [("ou", 2), ("stiff-gbm", 3)]:
            _, calls, _ = self.invoke([name])
            self.assertEqual(len(calls), count + 1)
            _, calls, _ = self.invoke([name, "--method", "ees25"])
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0].args[0][-2:], ["--method", "ees25"])

    def test_training_allocation_keeps_benchmark_and_user_settings(self):
        with patch.dict(launcher.os.environ, {}, clear=True):
            _, calls, _ = self.invoke(["kuramoto"])
            self.assertEqual(
                calls[0].kwargs["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"], "false"
            )
            _, calls, _ = self.invoke(["sphere"])
            self.assertEqual(
                calls[0].kwargs["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"], "false"
            )
        with patch.dict(launcher.os.environ, {"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}):
            _, calls, _ = self.invoke(["kuramoto"])
            self.assertEqual(
                calls[0].kwargs["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"], "true"
            )

    def test_comparison_streams_failure_and_preserves_log(self):
        path = REPO / "experiments/kuramoto/scripts/run_comparison.py"
        spec = importlib.util.spec_from_file_location("comparison", path)
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
        _run_cell = comparison._run_cell
        from experiments.kuramoto.experiment.config import ExperimentConfig, Solvers

        config = ExperimentConfig(
            N=3, batch_size=2, epochs=1, solver=Solvers.CG2, adjoint="checkpoint_full"
        )
        output_dir = self.root / "cg2_full_seed0"
        real_popen = subprocess.Popen

        def child(cmd, **kwargs):
            self.assertEqual(cmd[cmd.index("--index") + 1], "2")
            self.assertNotIn("--n-steps", cmd)
            self.assertFalse(
                any(output_dir.iterdir()),
                "Child must receive an empty output directory",
            )
            return real_popen(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "print('visible failure'); raise SystemExit(7)",
                ],
                **kwargs,
            )

        terminal = io.StringIO()
        with (
            contextlib.redirect_stdout(terminal),
            patch("subprocess.Popen", side_effect=child),
        ):
            result = _run_cell(self.root / "config.toml", 2, config, output_dir)
        self.assertEqual(result["exit_code"], 7)
        self.assertIn("visible failure", terminal.getvalue())
        self.assertIn("visible failure", (output_dir / "train.log").read_text())


if __name__ == "__main__":
    unittest.main()
