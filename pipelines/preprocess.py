
from loguru import logger

from patent.cli import clean_data, embed_data, reserialize_data
from patent.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from pathlib import Path


def main():
    logger.info("Starting preprocessing pipeline...")

    snapshot_file = RAW_DATA_DIR / "arxiv-metadata-oai-snapshot.json.zst"
    updates_dir = RAW_DATA_DIR / "updates"

    targets = []

    if snapshot_file.exists():
        targets.append((snapshot_file, True))

    if updates_dir.exists() and updates_dir.is_dir():
        for update_subdir in updates_dir.iterdir():
            if update_subdir.is_dir():
                targets.append((update_subdir, False))

    if not targets:
        logger.warning(f"No raw data found in {RAW_DATA_DIR}. Exiting.")
        return

    for raw_path, is_json in targets:
        logger.info(f"=== Processing source: {raw_path.name} ===")

        # Strip .zst suffix for clean output naming
        stem_name = raw_path.name
        if stem_name.endswith(".zst"):
            stem_name = stem_name[:-4]
        out_name = (
            f"{stem_name}.parquet"
            if raw_path.is_dir()
            else Path(stem_name).with_suffix(".parquet").name
        )

        serialized_path = INTERIM_DATA_DIR / "serialized" / out_name
        cleaned_path = INTERIM_DATA_DIR / "cleaned" / out_name
        processed_path = PROCESSED_DATA_DIR / out_name

        # 1. Serialize
        if not serialized_path.exists():
            logger.info(f"Serializing {raw_path}...")
            reserialize_data(file_path=raw_path, output_path=serialized_path, is_json=is_json)
        else:
            logger.info(
                f"Skipping serialize for {raw_path.name} -> {serialized_path.name} already exists."
            )

        # 2. Clean
        if not cleaned_path.exists():
            logger.info(f"Cleaning {serialized_path.name}...")
            clean_data(file_path=serialized_path, output_path=cleaned_path)
        else:
            logger.info(
                f"Skipping clean for {serialized_path.name} -> {cleaned_path.name} already exists."
            )

        # 3. Embed
        if not processed_path.exists():
            logger.info(f"Embedding {cleaned_path.name}...")
            embed_data(
                file_path=cleaned_path,
                output_path=processed_path,
                embedder_spec="sentence-transformers:all-MiniLM-L6-v2",
                batch_size=50000,
            )
        else:
            logger.info(
                f"Skipping embed for {cleaned_path.name} -> {processed_path.name} already exists."
            )

    logger.success("Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()
