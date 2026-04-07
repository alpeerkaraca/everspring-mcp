"""Run scrape jobs from config\\module_submodule_urls.csv with parallel workers.

Defaults:
- 5 parallel jobs
- 5 scrape concurrency per job
- reference lane -> spring-docs/raw-data
- api-doc lane -> spring-docs/raw-data
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("run_scrape_matrix")


@dataclass(frozen=True)
class ScrapeJob:
    module: str
    submodule: str | None
    version: str
    content_type: str
    entry_url: str
    s3_prefix: str


@dataclass(frozen=True)
class JobResult:
    job: ScrapeJob
    exit_code: int
    log_path: Path
    command: list[str]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "job"


def _normalize_prefix(value: str) -> str:
    """Normalize S3 prefixes to avoid leading/trailing slash mismatches."""
    normalized = value.strip().strip("/")
    if not normalized:
        raise ValueError("S3 prefix cannot be empty")
    return normalized


def build_jobs(
    csv_path: Path,
    include_reference: bool,
    include_api: bool,
    reference_prefix: str,
    api_prefix: str,
) -> list[ScrapeJob]:
    reference_prefix = _normalize_prefix(reference_prefix)
    api_prefix = _normalize_prefix(api_prefix)

    jobs: list[ScrapeJob] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            module = (row.get("module") or "").strip()
            submodule = (row.get("submodule") or "").strip() or None
            version = (row.get("version") or "").strip()
            reference_url = (row.get("reference_url") or "").strip()
            api_url = (row.get("api_url") or "").strip()

            if not module or not version:
                continue

            if include_reference and reference_url:
                jobs.append(
                    ScrapeJob(
                        module=module,
                        submodule=submodule,
                        version=version,
                        content_type="reference",
                        entry_url=reference_url,
                        s3_prefix=reference_prefix,
                    )
                )

            if include_api and api_url:
                jobs.append(
                    ScrapeJob(
                        module=module,
                        submodule=submodule,
                        version=version,
                        content_type="api-doc",
                        entry_url=api_url,
                        s3_prefix=api_prefix,
                    )
                )

    return jobs


def build_command(
    job: ScrapeJob,
    repo_root: Path,
    scrape_concurrency: int,
    uv_bin: str,
) -> list[str]:
    cmd = [
        uv_bin,
        "run",
        "python",
        "-m",
        "everspring_mcp.main",
        "scrape",
        "--entry-url",
        job.entry_url,
        "--module",
        job.module,
        "--version",
        job.version,
        "--content-type",
        job.content_type,
        "--concurrency",
        str(scrape_concurrency),
        "--s3-prefix",
        job.s3_prefix,
    ]
    if job.submodule:
        cmd.extend(["--submodule", job.submodule])
    return cmd


def run_job(
    job: ScrapeJob,
    repo_root: Path,
    log_dir: Path,
    scrape_concurrency: int,
    uv_bin: str,
) -> JobResult:
    sub = job.submodule or "root"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{_safe_name(job.module)}_{_safe_name(sub)}_{_safe_name(job.content_type)}_{stamp}.log"
    log_path = log_dir / log_name
    cmd = build_command(job, repo_root, scrape_concurrency, uv_bin)

    exit_code = 1
    try:
        with log_path.open("w", encoding="utf-8") as log_fh:
            log_fh.write(f"COMMAND: {' '.join(cmd)}\n\n")
            logger.info(
                "Job: %s/%s [%s] -> %s",
                job.module,
                sub,
                job.content_type,
                log_path,
            )
            completed = subprocess.run(
                cmd,
                cwd=repo_root,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            exit_code = completed.returncode
    except FileNotFoundError:
        with log_path.open("a", encoding="utf-8") as log_fh:
            log_fh.write(f"\nERROR: Executable not found: {uv_bin}\n")
        exit_code = 127

    return JobResult(job=job, exit_code=exit_code, log_path=log_path, command=cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scrape matrix from CSV with parallel workers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default="config\\module_submodule_urls.csv",
        help="Path to CSV file.",
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=5,
        help="Number of scrape jobs to run in parallel.",
    )
    parser.add_argument(
        "--scrape-concurrency",
        type=int,
        default=5,
        help="--concurrency passed to each scrape command.",
    )
    parser.add_argument(
        "--include",
        choices=["both", "reference", "api"],
        default="both",
        help="Which lanes to execute.",
    )
    parser.add_argument(
        "--reference-prefix",
        default="spring-docs/raw-data",
        help="S3 prefix for reference documentation lane.",
    )
    parser.add_argument(
        "--api-prefix",
        default="spring-docs/raw-data",
        help="S3 prefix for API documentation lane.",
    )
    parser.add_argument(
        "--uv-bin",
        default="uv",
        help="uv executable name or path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = (repo_root / args.csv).resolve()
    log_dir = repo_root / "logs" / "scrape-matrix"
    log_dir.mkdir(parents=True, exist_ok=True)

    include_reference = args.include in ("both", "reference")
    include_api = args.include in ("both", "api")

    try:
        jobs = build_jobs(
            csv_path=csv_path,
            include_reference=include_reference,
            include_api=include_api,
            reference_prefix=args.reference_prefix,
            api_prefix=args.api_prefix,
        )
    except ValueError as exc:
        logger.error("Invalid scrape matrix configuration: %s", exc)
        return 2

    if not jobs:
        logger.info("No jobs found in %s", csv_path)
        return 0

    logger.info("Loaded %d jobs from %s", len(jobs), csv_path)
    logger.info(
        "Running with %d parallel jobs and %d scrape concurrency per job.",
        args.parallel_jobs,
        args.scrape_concurrency,
    )

    if args.dry_run:
        for job in jobs:
            cmd = build_command(job, repo_root, args.scrape_concurrency, args.uv_bin)
            logger.info("%s", " ".join(cmd))
        return 0

    failures: list[JobResult] = []
    successes = 0
    with ThreadPoolExecutor(max_workers=args.parallel_jobs) as executor:
        futures = {
            executor.submit(
                run_job,
                job,
                repo_root,
                log_dir,
                args.scrape_concurrency,
                args.uv_bin,
            ): job
            for job in jobs
        }

        for future in as_completed(futures):
            result = future.result()
            lane = f"{result.job.module}/{result.job.submodule or 'root'} [{result.job.content_type}]"
            if result.exit_code == 0:
                successes += 1
                logger.info("[OK]   %s -> %s", lane, result.log_path)
            else:
                failures.append(result)
                logger.error(
                    "[FAIL] %s (exit %d) -> %s",
                    lane,
                    result.exit_code,
                    result.log_path,
                )

    logger.info("Completed: %d succeeded, %d failed.", successes, len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
