#!/usr/bin/env python3
"""
Resumable wrapper around `shamela-ingest`.

First run:  discovers all book folders under BOOKS_ROOT, writes them to
            remaining.txt, then starts processing.
Later runs: picks up where it left off by reading remaining.txt.

Each book that completes successfully is removed from remaining.txt immediately,
so a crash/Ctrl-C loses at most one book's work. The underlying LanceDB writes
upsert (merge-insert keyed on chunk_id), so re-running a book — whether it failed
partway or completed — replaces its chunks rather than duplicating them. Retries
are therefore safe and idempotent.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

BOOKS_ROOT = Path(
    r"C:\Users\Ali\.cache\huggingface\hub\datasets--AuthenticIlm--Shamela4_Full_DB"
    r"\snapshots\07554bee488a12955dd5231d08487ae7ce767d1e",
    r"17__الفقه-الحنبلي"
)

INGEST_BIN = "shamela-ingest"
PROJECT_DIR = Path(__file__).parent

# DB dir and resume file are overridable via env so a fresh run (e.g. to validate
# a chunking change) can target a NEW database without clobbering the existing
# one or its resume state. Defaults preserve the original behavior.
LANCEDB_DIR = os.environ.get("INGEST_LANCEDB", "lancedb")
REMAINING_FILE = Path(
    os.environ.get("INGEST_REMAINING", str(PROJECT_DIR / "remaining.txt"))
)

# Flags forwarded to every shamela-ingest invocation.
# Adjust these for your run (e.g. remove --mock-embeddings when the GPU
# embedder is live, remove --skip-enrichment when you want Claude metadata).
EXTRA_FLAGS = [
    # "--skip-enrichment",
    # "--mock-embeddings",
]


def discover_books(root: Path) -> list[str]:
    """Find every directory containing book_metadata.json, recursively."""
    books = []
    for dirpath, _, filenames in os.walk(root):
        if "book_metadata.json" in filenames:
            books.append(dirpath)
    books.sort()
    return books


def init_remaining():
    """Create remaining.txt from a full discovery scan."""
    print(f"Discovering books under {BOOKS_ROOT} ...")
    books = discover_books(BOOKS_ROOT)
    print(f"Found {len(books)} books. Writing remaining.txt")
    REMAINING_FILE.write_text("\n".join(books) + "\n", encoding="utf-8")
    return books


def load_remaining() -> list[str]:
    """Read remaining.txt, returning non-empty lines."""
    text = REMAINING_FILE.read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def save_remaining(books: list[str]):
    """Atomically rewrite remaining.txt."""
    tmp = REMAINING_FILE.with_suffix(".tmp")
    tmp.write_text("\n".join(books) + "\n", encoding="utf-8")
    tmp.replace(REMAINING_FILE)


def ingest_one(book_dir: str) -> bool:
    """Run shamela-ingest on a single book directory. Returns True on success."""
    cmd = [
        "cargo", "run", "--release", "--bin", INGEST_BIN, "--",
        "--input", book_dir,
        "--lancedb", LANCEDB_DIR,
        *EXTRA_FLAGS,
    ]
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode == 0


def main():
    if not REMAINING_FILE.exists():
        books = init_remaining()
    else:
        books = load_remaining()
        print(f"Resuming: {len(books)} books remaining")

    if not books:
        print("Nothing to do — remaining.txt is empty.")
        return

    done = 0
    failed = 0
    total = len(books)
    t0 = time.time()

    for i, book_dir in enumerate(list(books)):
        elapsed = time.time() - t0
        print(f"\n[{i+1}/{total}] {Path(book_dir).name}  (done={done} failed={failed} elapsed={elapsed:.0f}s)")

        ok = ingest_one(book_dir)
        if ok:
            done += 1
            books.remove(book_dir)
            save_remaining(books)
        else:
            failed += 1
            # Leave it in remaining.txt for a future retry.
            print(f"  FAILED — left in remaining.txt for retry")

    elapsed = time.time() - t0
    print(f"\nFinished: {done} succeeded, {failed} failed, {len(books)} remaining, {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
