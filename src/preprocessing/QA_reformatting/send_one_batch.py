#!/usr/bin/env python3
import os, sys, time, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from openai import OpenAI

MANIFEST_NAME = "manifest.csv"

def find_prepared_dir(batch_file: Path) -> Path:
    """
    Infer prepared_batches dir from a file like:
      data/step_1/prepared_batches/train_batch_007.jsonl
    """
    # .../prepared_batches/<file>
    pb = batch_file.parent
    if pb.name != "prepared_batches":
        # Try to locate prepared_batches above
        raise FileNotFoundError(
            f"Expected file under a 'prepared_batches' directory, got: {batch_file}"
        )
    return pb

def load_manifest(prepared_dir: Path) -> pd.DataFrame:
    mpath = prepared_dir / MANIFEST_NAME
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")
    return pd.read_csv(mpath)

def save_manifest(prepared_dir: Path, df: pd.DataFrame):
    (prepared_dir / MANIFEST_NAME).write_text(df.to_csv(index=False), encoding="utf-8")

def parse_kind_batchnum(file_path: Path):
    """
    From filename like train_batch_007.jsonl -> ('train', 7)
    or val_batch_018.jsonl -> ('val', 18)
    """
    stem = file_path.stem  # e.g., train_batch_007
    # expected pattern <kind>_batch_<NNN>
    parts = stem.split("_batch_")
    if len(parts) != 2:
        return None, None
    kind = parts[0]
    try:
        batch_num = int(parts[1])
    except ValueError:
        # might be zero-padded string like 007 - int handles that
        batch_num = None
    return kind, batch_num

def wait_until_not_enqueued(client, batch_id: str, poll_secs: int = 10, timeout_secs: int = 1200):
    """
    Wait until batch is no longer 'validating' or 'enqueued'.
    Returns final observed status when leaving those states (e.g., 'in_progress', 'completed', 'failed', 'cancelled', 'expired').
    """
    print("    Waiting for batch to leave queue (validating/enqueued)...", end="", flush=True)
    start = time.time()
    while True:
        b = client.batches.retrieve(batch_id)
        st = b.status
        if st not in ("validating", "enqueued"):
            print(f" ✓ {st}")
            return st, b
        if time.time() - start > timeout_secs:
            print(" (timeout) — continuing.")
            return st, b
        time.sleep(poll_secs)
        print(".", end="", flush=True)

def safe_submit_with_backoff(client, file_path: Path, description: str, max_retries: int = 8):
    """
    Submit with exponential backoff if queue/ratelimit error occurs.
    """
    delay = 10
    for attempt in range(1, max_retries + 1):
        try:
            with open(file_path, "rb") as f:
                up = client.files.create(file=f, purpose="batch")
            return client.batches.create(
                input_file_id=up.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": description}
            )
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["limit", "rate", "too many", "enqueued", "2m"]):
                print(f"    Submit blocked (attempt {attempt}): {e}")
                print(f"    Sleeping {delay}s before retry...")
                time.sleep(delay)
                delay = min(delay * 2, 300)
                continue
            raise

def main():
    if len(sys.argv) < 2:
        print("Usage: python submit_one.py path/to/prepared_batches/<train|val>_batch_###.jsonl [--desc 'optional description']")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    # parse args
    file_path = Path(sys.argv[1]).resolve()
    if not file_path.exists():
        print(f"Error: file not found: {file_path}")
        sys.exit(1)
    desc = " ".join(sys.argv[2:]).replace("--desc", "").strip() or f"Manual submit {file_path.name}"

    # derive dirs and manifest
    prepared_dir = find_prepared_dir(file_path)
    manifest = load_manifest(prepared_dir)

    kind, batch_num = parse_kind_batchnum(file_path)
    if not kind or not batch_num:
        print(f"Error: cannot parse kind/batch number from filename: {file_path.name}")
        sys.exit(1)

    # locate or create manifest row
    # try exact path match first
    idx_matches = manifest.index[manifest["file_path"] == str(file_path)].tolist()
    if not idx_matches:
        # fallback: try by kind+batch_num
        idx_matches = manifest.index[
            (manifest["kind"] == kind) & (manifest["batch_num"] == batch_num)
        ].tolist()

    if idx_matches:
        idx = idx_matches[0]
    else:
        # row not in manifest — create one (minimal)
        print("Warning: batch file not present in manifest; adding a new row.")
        new_row = {
            "kind": kind,
            "batch_num": batch_num,
            "file_path": str(file_path),
            "num_cases": "",
            "est_input_tokens": "",
            "status": "PENDING",
            "batch_id": "",
            "submitted_at": "",
            "started_at": "",
            "completed_at": ""
        }
        manifest = pd.concat([manifest, pd.DataFrame([new_row])], ignore_index=True)
        idx = manifest.index[-1]

    client = OpenAI(api_key=api_key)

    print(f"Submitting: {file_path.name} ({kind} batch {batch_num})")
    batch = safe_submit_with_backoff(client, file_path, desc)
    print(f"Submitted batch_id={batch.id} status={batch.status}")

    # update manifest after submit
    manifest.loc[idx, "batch_id"] = batch.id
    manifest.loc[idx, "status"] = "SUBMITTED"
    manifest.loc[idx, "submitted_at"] = datetime.utcnow().isoformat()
    save_manifest(prepared_dir, manifest)

    # wait until it leaves the queue and update status
    status, full = wait_until_not_enqueued(client, batch.id)
    now_iso = datetime.utcnow().isoformat()
    if status == "in_progress":
        manifest.loc[idx, "status"] = "IN_PROGRESS"
        manifest.loc[idx, "started_at"] = now_iso
    elif status in ("completed", "failed", "cancelled", "expired"):
        manifest.loc[idx, "status"] = status.upper()
        manifest.loc[idx, "completed_at"] = now_iso
    else:
        # still validating/enqueued after timeout — mark as queued:<status>
        manifest.loc[idx, "status"] = f"QUEUED:{status}"
    save_manifest(prepared_dir, manifest)

    print("batch complete; manifest updated")
    print(f"  {prepared_dir / MANIFEST_NAME}")

if __name__ == "__main__":
    main()
