"""CPU diagnostic: unchanged screens, independent oracle, learned baselines.

Original split is a development comparison. Fresh split is confirmation of this
fixed implementation, not external biological validation. No threshold tuning.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

from cardiagent.conditional import ConditionalQuantileGenerator
from cardiagent.generator import ChallengeGenerator
from cardiagent.ml import AgentGeneratorModel
from cardiagent.serialization import write_json


class ReferenceControl:
    def __init__(self, seed):
        self.generator = ChallengeGenerator(seed=seed + 1000000)
        self.counter = 0

    def sample(self, *, domain, severity, difficulty, count, agent_id_prefix):
        rows = []
        for _ in range(count):
            self.counter += 1
            rows.append(
                self.generator.generate(
                    domain,
                    severity=severity,
                    difficulty=difficulty,
                    agent_id=f"{agent_id_prefix}-{self.counter}",
                )
            )
        return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation/cpu/recovery"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_paths = sorted((root / "src/cardiagent").glob("*.py")) + [
        root / "validation/colab/validation_colab.py",
        Path(__file__),
    ]
    source_at_start = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
    }
    os.environ["CARDIAGENT_VALIDATION_OUTPUT"] = str(args.output / "scratch")
    spec = importlib.util.spec_from_file_location(
        "heldout_protocol", root / "validation/colab/validation_colab.py"
    )
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    import torch

    torch.set_num_threads(1)
    from cardiagent.screen_calibration import ks_null_gate

    report = {
        "ks_null_calibration": ks_null_gate(9, 9, h.SCREEN["max_ks_d"]),
        "environment": h.environment_report(),
        "schema_version": "cardiagent-generator-diagnosis-v1",
        "cpu_only": True,
        "screen": h.SCREEN,
        "model_seeds": list(h.MODEL_SEEDS),
        "epochs": h.ML_EPOCHS,
        "samples_per_condition": 9,
        "original_failures_preserved": "validation/cpu/heldout",
        "scientific_scope": "internal synthetic source; no patient data",
        "protocols": {},
    }
    for name, data_seed, split_seed in [
        ("development", 20260917, 20260918),
        ("fresh_confirmation", 20261008, 20261009),
    ]:
        print("Protocol", name, flush=True)
        h.DATA_SEED = data_seed
        h.SPLIT_SEED = split_seed
        train, validation, test, manifest = h.split_population(h.generate_population())
        assert h.verify_split_integrity(train, validation, test)["status"] == "PASS"
        result = {
            "data_seed": data_seed,
            "split_seed": split_seed,
            "split_manifest_sha256": h.sha256_json(manifest),
            "families": {},
        }
        for family in (
            "independent_reference_control",
            "support_aware_cvae",
            "conditional_quantile",
        ):
            runs = []
            times = []
            for seed in h.MODEL_SEEDS:
                started = time.monotonic()
                if family == "independent_reference_control":
                    model = ReferenceControl(seed)
                elif family == "support_aware_cvae":
                    model = AgentGeneratorModel(seed=seed).fit(
                        train,
                        epochs=h.ML_EPOCHS,
                        batch_size=h.BATCH_SIZE,
                        learning_rate=2e-3,
                        beta=0.015,
                    )
                else:
                    model = ConditionalQuantileGenerator(seed=seed).fit(train)
                runs.append(h.evaluate_model(model, seed, train, validation, test))
                times.append(time.monotonic() - started)
                print(name, family, seed, "seconds", round(times[-1], 1), flush=True)
            result["families"][family] = {
                "screen": h.aggregate_screen(runs),
                "runs": runs,
                "elapsed_seconds": times,
            }
        report["protocols"][name] = result
    report["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "src/cardiagent").glob("*.py"))
    }
    report["harness_sha256"] = hashlib.sha256(
        (root / "validation/colab/validation_colab.py").read_bytes()
    ).hexdigest()
    report["diagnostic_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if source_at_start != {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
    }:
        raise RuntimeError("Source changed during validation; rerun a stable checkout")
    report = h.canonical(report)
    report["undefined_numeric_values"] = (
        "Undefined correlations are JSON null, never zero. Pair lists remain explicit."
    )
    args.output.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(report, sort_keys=True, allow_nan=False).encode()
    (args.output / "results.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    summary = {k: v for k, v in report.items() if k != "protocols"}
    summary["protocols"] = {
        name: {
            "data_seed": p["data_seed"],
            "split_seed": p["split_seed"],
            "split_manifest_sha256": p["split_manifest_sha256"],
            "families": {
                family: {
                    "screen": f["screen"],
                    "domain_auc": [
                        m["discriminator"]["mean_auc"]
                        for r in f["runs"]
                        for m in r["domain_aggregate"].values()
                    ],
                    "elapsed_seconds": f["elapsed_seconds"],
                }
                for family, f in p["families"].items()
            },
        }
        for name, p in report["protocols"].items()
    }
    write_json(summary, args.output / "summary.json")
    write_json(
        {
            "results.json.gz": hashlib.sha256(
                (args.output / "results.json.gz").read_bytes()
            ).hexdigest(),
            "uncompressed_results_sha256": hashlib.sha256(raw).hexdigest(),
        },
        args.output / "checksums.json",
    )
    print(
        json.dumps(
            {
                n: {f: r["screen"]["status"] for f, r in p["families"].items()}
                for n, p in summary["protocols"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
