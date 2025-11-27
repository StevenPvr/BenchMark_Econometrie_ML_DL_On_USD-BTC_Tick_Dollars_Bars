#!/usr/bin/env python3
"""Consolidate partitioned parquet datasets into a single file for better performance."""

from pathlib import Path
import pandas as pd  # type: ignore[import-untyped]
from datetime import datetime
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_logger

logger = get_logger(__name__)


def create_partitioned_dataset_index(
    input_path: Path,
    index_file: Path | None = None
) -> dict:
    """Create an index of partitioned parquet files without loading data.

    This creates a lightweight index that data_preparation can use to process
    files individually instead of consolidating everything.

    Args:
        input_path: Directory containing partitioned parquet files
        index_file: Optional path to save the index JSON

    Returns:
        Dictionary with file information
    """
    if not input_path.is_dir():
        raise ValueError(f"Input path must be a directory: {input_path}")

    parquet_files = list(input_path.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {input_path}")

    index = {
        "input_directory": str(input_path),
        "files": [],
        "total_files": len(parquet_files),
        "created_at": pd.Timestamp.now().isoformat()
    }

    logger.info(f"📁 Indexing {len(parquet_files)} parquet files...")

    for file_path in sorted(parquet_files):
        try:
            # Get file stats without loading data
            stat = file_path.stat()
            index["files"].append({
                "path": str(file_path),
                "name": file_path.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2)
            })
        except Exception as e:
            logger.warning(f"Could not index {file_path}: {e}")

    # Save index if requested
    if index_file:
        index_file.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        logger.info(f"💾 Index saved to {index_file}")

    logger.info(f"✅ Indexed {len(index['files'])} files")
    return index


# REMOVED: Manual consolidation functions - use automatic processing instead


# REMOVED: Auto-processing moved to data_fetching/main.py for cleaner separation


def main():
    """Main script - data consolidation utilities."""
    logger.info("🔄 Data consolidation utilities...")

    # Note: Automatic processing has been moved to data_fetching/main.py
    # This script now only provides utility functions for manual consolidation
    logger.info("ℹ️  For automatic data processing, use: python -m src.data_fetching.main")
    logger.info("ℹ️  For manual consolidation, use the utility functions in this module")

    # You can add command-line arguments here for manual operations
    # For now, just show available functions
    logger.info("✅ Script loaded successfully - use functions like create_partitioned_dataset_index()")
    sys.exit(0)


if __name__ == "__main__":
    main()
