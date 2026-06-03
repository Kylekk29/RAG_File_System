import glob
import hashlib
import json
import os
import re
import shutil
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import config

logger = logging.getLogger(__name__)


class BookVersionManager:
    def __init__(self):
        logger.info("Initializing BookVersionManager...")
        self.metadata_dir = config.METADATA_DIR
        self.archive_dir = config.ARCHIVE_DIR
        logger.info(f"  Metadata dir: {self.metadata_dir}")
        logger.info(f"  Archive dir: {self.archive_dir}")

    @staticmethod
    def clean_id(raw: str) -> str:
        name = os.path.splitext(os.path.basename(raw))[0]
        cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned:
            hash_int = int(hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest(), 16) % 100000
            return f"book_{hash_int}"
        if cleaned != name:
            short_hash = hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest()[:6]
            cleaned = f"{cleaned}_{short_hash}"
        return cleaned

    def calculate_file_hash(self, file_path: str) -> str:
        file_size = os.path.getsize(file_path)
        logger.info("Calculating hash for: %s (%d bytes)", os.path.basename(file_path), file_size)
        start_time = time.time()
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()
        elapsed = time.time() - start_time
        logger.info("  Hash: %s..., calculated in %.3fs (%.1f MB/s)",
                     file_hash[:16], elapsed,
                     file_size / (1024 * 1024) / elapsed if elapsed > 0 else 0)
        return file_hash

    def _meta_path(self, book_identifier: str) -> str:
        safe_id = book_identifier.replace("/", "_").replace("\\", "_")
        return os.path.join(self.metadata_dir, f"{safe_id}_book_meta.json")

    def _archive_path(self, book_identifier: str, file_hash: str, ext: str) -> str:
        safe_id = book_identifier.replace("/", "_").replace("\\", "_")
        return os.path.join(self.archive_dir, f"{safe_id}_{file_hash[:16]}{ext}")

    def load_metadata(self, book_identifier: str) -> Dict:
        path = self._meta_path(book_identifier)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info(
                f"Loaded metadata for {book_identifier}: "
                f"{len(metadata.get('versions', []))} versions"
            )
            return metadata
        logger.info(f"No existing metadata for {book_identifier}")
        return {
            "book_name": book_identifier,
            "versions": [],
            "current_hash": None,
            "current_db_id": None,
        }

    def save_metadata(self, book_identifier: str, metadata: Dict) -> None:
        path = self._meta_path(book_identifier)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata for {book_identifier}")

    def check_version(
        self, file_path: str, book_identifier: str, base_dir: Optional[str] = None,
    ) -> Dict:
        logger.info(f"Checking version for: {book_identifier}")

        start_time = time.time()
        file_hash = self.calculate_file_hash(file_path)
        metadata = self.load_metadata(book_identifier)

        for version in metadata["versions"]:
            if version["hash"] == file_hash:
                db_id = version["db_id"]
                logger.info(
                    f"  DUPLICATE FOUND (v{version['version']}, "
                    f"from {version['timestamp'][:10]})"
                )
                logger.info(f"  Reusing DB ID: {db_id}")

                if base_dir is not None:
                    index_file = os.path.join(base_dir, db_id, "index.faiss")
                    if not os.path.exists(index_file):
                        logger.warning(
                            f"  FAISS index missing on disk for db_id={db_id} "
                            f"- forcing rebuild (action=repaired)"
                        )
                        _, ext = os.path.splitext(file_path)
                        return {
                            "is_new": True,
                            "file_hash": file_hash,
                            "db_id": db_id,
                            "action": "repaired",
                            "version_number": version["version"],
                            "message": (
                                f"Index missing on disk - rebuilding "
                                f"(v{version['version']}, db_id={db_id})"
                            ),
                            "_internal": {
                                "skip_commit": True,
                                "file_path": file_path,
                                "archive_path": self._archive_path(
                                    book_identifier, file_hash, ext
                                ),
                                "category": "book",
                                "is_update": True,
                            },
                        }

                return {
                    "is_new": False,
                    "file_hash": file_hash,
                    "db_id": db_id,
                    "action": "reused",
                    "version_number": version["version"],
                    "message": (
                        f"Identical content detected - reusing existing index "
                        f"(v{version['version']}, created {version['timestamp'][:10]})"
                    ),
                    "_internal": None,
                }

        logger.info("  NEW VERSION DETECTED")
        _, ext = os.path.splitext(file_path)
        version_number = len(metadata["versions"]) + 1
        db_id = f"{book_identifier}_{file_hash[:12]}"
        is_update = len(metadata["versions"]) > 0
        action = "updated" if is_update else "created"
        archive_path = self._archive_path(book_identifier, file_hash, ext)

        logger.info(f"  New version: v{version_number}")
        logger.info(f"  DB ID: {db_id}")

        msg = (
            f"New version registered (v{version_number}, DB: {db_id})"
            if not is_update
            else f"Updated to v{version_number} - previous version archived"
        )

        elapsed = time.time() - start_time
        logger.info(f"  Version check completed in {elapsed:.2f}s")

        return {
            "is_new": True,
            "file_hash": file_hash,
            "db_id": db_id,
            "action": action,
            "version_number": version_number,
            "message": msg,
            "_internal": {
                "skip_commit": False,
                "file_path": file_path,
                "archive_path": archive_path,
                "category": "book",
                "is_update": is_update,
            },
        }

    def commit_new_version(self, book_identifier: str, check_result: Dict) -> None:
        internal = check_result.get("_internal") or {}
        if internal.get("skip_commit"):
            logger.info(
                f"  commit_new_version: skipping (repaired case, "
                f"metadata already exists for {book_identifier})"
            )
            return

        file_path = internal["file_path"]
        archive_path = internal["archive_path"]

        if not os.path.exists(archive_path):
            logger.info(f"  Archiving to: {archive_path}")
            try:
                shutil.copy2(file_path, archive_path)
            except Exception as e:
                logger.error(f"  Failed to archive file: {e}")
                raise RuntimeError(f"Failed to archive file {file_path}: {e}")

        metadata = self.load_metadata(book_identifier)

        for v in metadata["versions"]:
            if v["hash"] == check_result["file_hash"]:
                logger.info(
                    f"  commit_new_version: version already present "
                    f"(concurrent commit?), skipping."
                )
                return

        version_entry = {
            "version": check_result["version_number"],
            "hash": check_result["file_hash"],
            "db_id": check_result["db_id"],
            "timestamp": datetime.now().isoformat(),
            "archive_path": archive_path,
            "category": internal.get("category", "book"),
        }

        metadata["versions"].append(version_entry)
        metadata["current_hash"] = check_result["file_hash"]
        metadata["current_db_id"] = check_result["db_id"]
        logger.debug("Version count after append: %d (max=%d)", len(metadata["versions"]), config.MAX_VERSIONS_PER_FILE)

        pruned = 0
        while len(metadata["versions"]) > config.MAX_VERSIONS_PER_FILE:
            old = metadata["versions"].pop(0)
            old_archive = old.get("archive_path", "")
            logger.info("  Pruning old version: v%d (%s...)", old["version"], old["hash"][:12])
            if old_archive and os.path.exists(old_archive):
                try:
                    os.remove(old_archive)
                    logger.info("  Removed old archive: %s", old_archive)
                    pruned += 1
                except OSError as e:
                    logger.warning("  Could not remove old archive %s: %s", old_archive, e)

        try:
            self.save_metadata(book_identifier, metadata)
        except Exception as e:
            logger.error(f"  Failed to save metadata: {e}")
            raise RuntimeError(f"Failed to save version metadata: {e}")

        logger.info(
            f"  Version committed: v{check_result['version_number']} "
            f"for {book_identifier}"
        )

    def get_current_db_id(self, book_identifier: str) -> Optional[str]:
        db_id = self.load_metadata(book_identifier).get("current_db_id")
        if db_id:
            logger.info(f"Current DB ID for {book_identifier}: {db_id}")
        else:
            logger.warning(f"No current DB ID for {book_identifier}")
        return db_id

    def get_version_history(self, book_identifier: str, book_name: str = None) -> List[Dict]:
        logger.info(f"Retrieving version history for: {book_identifier}")
        history = self.load_metadata(book_identifier)["versions"]
        if not history and book_name:
            for meta_file in glob.glob(
                os.path.join(self.metadata_dir, "*_book_meta.json")
            ):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("book_name") == book_name or meta.get("book_name") == book_identifier:
                        history = meta.get("versions", [])
                        break
                except (json.JSONDecodeError, OSError):
                    continue
        logger.info(f"  Found {len(history)} versions")
        return history

    def delete_book_metadata(self, book_identifier: str) -> None:
        path = self._meta_path(book_identifier)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted metadata for {book_identifier}")
            return
        for meta_file in glob.glob(
            os.path.join(self.metadata_dir, "*_book_meta.json")
        ):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("book_name") == book_identifier:
                    os.remove(meta_file)
                    logger.info(f"Deleted metadata by name: {meta_file}")
                    return
            except (json.JSONDecodeError, OSError):
                continue
