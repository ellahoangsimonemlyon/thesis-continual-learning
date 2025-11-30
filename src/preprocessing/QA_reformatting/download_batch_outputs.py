#!/usr/bin/env python3
import os, sys, json, time
from pathlib import Path
from datetime import datetime
import pandas as pd
from openai import OpenAI

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_manifest(prepared_dir: Path) -> pd.DataFrame:
    mpath = prepared_dir / "manifest.csv"
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")
    return pd.read_csv(mpath)

def save_manifest(prepared_dir: Path, df: pd.DataFrame):
    (prepared_dir / "manifest.csv").write_text(df.to_csv(index=False), encoding="utf-8")

def download_file_content(client, file_id: str) -> bytes:
    # OpenAI SDK: files.content returns a streaming Response. Use .read() on iterator.
    content = client.files.content(file_id)
    # content is a httpx.Response-like; return bytes
    return content.read()

def parse_and_save_clean_qa(output_jsonl_path: Path, clean_jsonl_path: Path):
    """
    Reads the raw batch output jsonl and extracts the model's JSON reply for each line.
    Assumes /v1/chat/completions responses with response.body.choices[0].message.content
    which itself should be a JSON object {"question": "...", "answer": "..."}.
    """
    out = []
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Successful responses have a "response" object
                resp = obj.get("response", {})
                body = resp.get("body", {})
                choices = body.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    # content should be a JSON object string per your prompt response_format
                    try:
                        qa = json.loads(content)
                        if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                            out.append(qa)
                    except Exception:
                        # If it's not JSON, skip or store a fallback
                        pass
            except Exception:
                pass
    if out:
        with open(clean_jsonl_path, "w", encoding="utf-8") as w:
            for qa in out:
                w.write(json.dumps(qa, ensure_ascii=False) + "\n")
    return len(out)

def main():
    if len(sys.argv) < 3 and "--step" not in sys.argv:
        print("Usage: python download_batch_outputs.py --step <number> [--parse]")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True)
    parser.add_argument("--parse", action="store_true", help="Also create a cleaned Q/A JSONL per batch")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    step_dir = Path(f"data/step_{args.step}")
    prepared_dir = step_dir / "prepared_batches"
    outputs_dir = prepared_dir / "outputs"
    ensure_dir(outputs_dir)

    manifest = read_manifest(prepared_dir)

    # Only rows with a batch_id get checked
    rows = manifest[manifest["batch_id"].astype(str).str.len() > 0].copy()
    if rows.empty:
        print(f"No batch_ids found in {prepared_dir}/manifest.csv — nothing to download yet.")
        sys.exit(0)

    for idx in rows.index:
        row = manifest.loc[idx]
        batch_id = row["batch_id"]
        kind = row["kind"]
        bnum = int(row["batch_num"])
        print(f"\nChecking {kind.upper()} batch {bnum} (batch_id={batch_id})…")

        try:
            b = client.batches.retrieve(batch_id)
        except Exception as e:
            print(f"retrieve error: {e}")
            continue

        print(f"  status: {b.status}")
        # If not completed, skip
        if b.status != "completed":
            continue

        # Download output file
        ofid = getattr(b, "output_file_id", None)
        if not ofid:
            print("No output_file_id on completed batch. Skipping.")
            manifest.loc[idx, "status"] = "COMPLETED_NO_OUTPUT"
            save_manifest(prepared_dir, manifest)
            continue

        out_path = outputs_dir / f"{kind}_batch_{bnum:03d}_output.jsonl"
        if out_path.exists():
            print(f"Output already exists: {out_path.name}")
        else:
            try:
                data = download_file_content(client, ofid)
                out_path.write_bytes(data)
                print(f"Saved: {out_path.name} ({len(data):,} bytes)")
            except Exception as e:
                print(f"download error: {e}")
                continue

        # Optional: parse to clean Q/A JSONL
        if args.parse:
            clean_path = outputs_dir / f"{kind}_batch_{bnum:03d}_qa.jsonl"
            n = parse_and_save_clean_qa(out_path, clean_path)
            print(f"Parsed Q/A: {n} items -> {clean_path.name}")

        # Update manifest timestamps/status
        manifest.loc[idx, "status"] = "COMPLETED"
        manifest.loc[idx, "completed_at"] = datetime.utcnow().isoformat()
        save_manifest(prepared_dir, manifest)

if __name__ == "__main__":
    main()
