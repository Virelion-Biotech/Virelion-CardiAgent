"""Execute the predeclared expanded CPU-only conditional quantile protocol."""

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
from cardiagent.models import ChallengeDomain
from cardiagent.screen_calibration import ks_null_gate
from cardiagent.serialization import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("validation/cpu/recovery/expanded"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "validation/cpu/recovery/expanded_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    source_paths = sorted((root / "src/cardiagent").glob("*.py")) + [
        root / "validation/colab/validation_colab.py",
        Path(__file__),
        protocol_path,
    ]
    fingerprints = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
    }
    os.environ["CARDIAGENT_VALIDATION_OUTPUT"] = str(args.output / "scratch")
    spec = importlib.util.spec_from_file_location(
        "quantile_protocol", root / "validation/colab/validation_colab.py"
    )
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)

    reference = ChallengeGenerator(protocol["reference_data_seed"])
    train = []
    test = []
    membership = {}
    ntrain = protocol["training_per_condition"]
    ntest = protocol["heldout_per_condition"]
    for domain in ChallengeDomain:
        for severity in h.SEVERITIES:
            key = f"{domain.value}|{severity:.1f}"
            training = [
                reference.generate(domain, severity=severity, difficulty=severity)
                for _ in range(ntrain)
            ]
            heldout = [
                reference.generate(domain, severity=severity, difficulty=severity)
                for _ in range(ntest)
            ]
            train.extend(training)
            test.extend(heldout)
            membership[key] = {
                "train_ids": [a.agent_id for a in training],
                "test_ids": [a.agent_id for a in heldout],
            }
    train_ids = {a.agent_id for a in train}
    test_ids = {a.agent_id for a in test}
    assert not train_ids & test_ids
    train_map = h.condition_map(train)
    test_map = h.condition_map(test)
    report = {
        "protocol": protocol,
        "protocol_sha256": fingerprints[str(protocol_path.relative_to(root))],
        "source_sha256": fingerprints,
        "environment": h.environment_report(),
        "screen": h.SCREEN,
        "ks_null_calibration": ks_null_gate(
            ntest, protocol["generated_per_condition"], h.SCREEN["max_ks_d"]
        ),
        "split_membership_sha256": h.sha256_json(membership),
        "results": [],
        "biological_validation": "not_validated",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in protocol["model_seeds"]:
        started = time.monotonic()
        model = ConditionalQuantileGenerator(seed).fit(train)
        run = {"model_seed": seed, "domain_aggregate": {}, "by_condition": {}}
        generated_by_domain = {domain: [] for domain in ChallengeDomain}
        for index, (key, train_rows) in enumerate(train_map.items()):
            domain_value, severity_text = key.split("|")
            domain = ChallengeDomain(domain_value)
            severity = float(severity_text)
            generated = model.sample(
                domain=domain,
                severity=severity,
                difficulty=severity,
                count=protocol["generated_per_condition"],
            )
            generated_by_domain[domain].extend(generated)
            metrics = h.aggregate_reference_metrics(
                test_map[key], generated, train_rows, test_map[key], seed_offset=seed * 1000 + index
            )
            run["by_condition"][key] = {"test": metrics, "screen": h.screen(metrics)}
        for domain in ChallengeDomain:
            train_rows = [a for a in train if a.domain == domain]
            test_rows = [a for a in test if a.domain == domain]
            metrics = h.aggregate_reference_metrics(
                test_rows,
                generated_by_domain[domain],
                train_rows,
                test_rows,
                seed_offset=seed * 2000 + len(domain.value),
                discriminator_seed=seed,
            )
            metrics["screen"] = h.screen(metrics)
            run["domain_aggregate"][domain.value] = metrics
        run["elapsed_seconds"] = time.monotonic() - started
        report["results"].append(run)
        checkpoint = args.output / f"quantile-seed-{seed}.json"
        model.save(checkpoint)
        expected = model.sample(
            domain=ChallengeDomain.ISCHEMIC, severity=0.5, difficulty=0.5, count=3
        )
        restored = ConditionalQuantileGenerator.load(checkpoint)
        assert [a.to_dict() for a in expected] == [
            a.to_dict()
            for a in restored.sample(
                domain=ChallengeDomain.ISCHEMIC, severity=0.5, difficulty=0.5, count=3
            )
        ]
        print(
            "Seed",
            seed,
            "seconds",
            round(run["elapsed_seconds"], 1),
            "screen",
            h.aggregate_screen([run])["status"],
            flush=True,
        )
    report["aggregate_screen"] = h.aggregate_screen(report["results"])
    if fingerprints != {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
    }:
        raise RuntimeError("Source changed during validation")
    raw = json.dumps(h.canonical(report), sort_keys=True, allow_nan=False).encode()
    (args.output / "results.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    (args.output / "split_membership.json.gz").write_bytes(
        gzip.compress(json.dumps(membership, sort_keys=True).encode(), mtime=0)
    )
    summary = {key: value for key, value in report.items() if key != "results"}
    summary["domain_auc"] = [
        m["discriminator"]["mean_auc"]
        for r in report["results"]
        for m in r["domain_aggregate"].values()
    ]
    summary["elapsed_seconds"] = [r["elapsed_seconds"] for r in report["results"]]
    write_json(summary, args.output / "summary.json")
    paths = list(args.output.glob("*.gz")) + list(args.output.glob("quantile-seed-*.json"))
    write_json(
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        args.output / "checksums.json",
    )
    print(report["aggregate_screen"], flush=True)
    return 0 if report["aggregate_screen"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
