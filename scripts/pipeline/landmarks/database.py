"""SQLite storage and bounded read-only queries for landmark packages."""

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

import numpy as np


SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class LandmarkBatch:
    map_xyz: np.ndarray
    descriptors: np.ndarray
    keypoints: np.ndarray
    terrain: np.ndarray
    tile_quality: np.ndarray


class LandmarkDatabaseWriter:
    def __init__(self, path, overwrite=False):
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"landmark database already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self._create_schema()

    def _create_schema(self):
        self.connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE tiles (
                tile_id INTEGER PRIMARY KEY, row INTEGER NOT NULL, col INTEGER NOT NULL,
                min_e REAL NOT NULL, min_n REAL NOT NULL, max_e REAL NOT NULL, max_n REAL NOT NULL,
                center_e REAL NOT NULL, center_n REAL NOT NULL, mean_elevation REAL,
                feature_count INTEGER NOT NULL, coverage REAL NOT NULL, mean_response REAL NOT NULL,
                quality REAL NOT NULL, UNIQUE(row, col)
            );
            CREATE INDEX tiles_row_col ON tiles(row, col);
            CREATE TABLE landmarks (
                landmark_id INTEGER PRIMARY KEY, tile_id INTEGER NOT NULL REFERENCES tiles(tile_id),
                east_m REAL NOT NULL, north_m REAL NOT NULL, up_m REAL NOT NULL,
                pixel_col REAL NOT NULL, pixel_row REAL NOT NULL, scale REAL NOT NULL,
                angle_deg REAL NOT NULL, response REAL NOT NULL, octave INTEGER NOT NULL,
                slope REAL NOT NULL, normal_e REAL NOT NULL, normal_n REAL NOT NULL, normal_u REAL NOT NULL,
                descriptor BLOB NOT NULL
            );
            CREATE INDEX landmarks_tile ON landmarks(tile_id);
        """)

    def write_metadata(self, metadata):
        rows = [(str(key), json.dumps(value, sort_keys=True)) for key, value in metadata.items()]
        self.connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", rows)

    def write_tile(self, tile):
        cursor = self.connection.execute("""
            INSERT INTO tiles(row, col, min_e, min_n, max_e, max_n, center_e, center_n,
                              mean_elevation, feature_count, coverage, mean_response, quality)
            VALUES (:row, :col, :min_e, :min_n, :max_e, :max_n, :center_e, :center_n,
                    :mean_elevation, :feature_count, :coverage, :mean_response, :quality)
        """, tile)
        return int(cursor.lastrowid)

    def write_landmarks(self, tile_id, rows):
        self.connection.executemany("""
            INSERT INTO landmarks(tile_id, east_m, north_m, up_m, pixel_col, pixel_row, scale,
                                  angle_deg, response, octave, slope, normal_e, normal_n, normal_u, descriptor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [(tile_id, *row) for row in rows])

    def close(self):
        self.connection.commit()
        self.connection.execute("PRAGMA optimize")
        self.connection.close()


class LandmarkDatabase:
    """Read-only, tile-bounded landmark access suitable for a companion computer."""
    def __init__(self, path):
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        self.connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
        self._metadata = {key: json.loads(value) for key, value in self.connection.execute(
            "SELECT key, value FROM metadata")}
        if self._metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported landmark database schema")

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def metadata(self):
        return dict(self._metadata)

    def _row_col_range(self, east_m, north_m, radius_m):
        if radius_m < 0:
            raise ValueError("radius_m must be non-negative")
        origin_e = float(self._metadata["grid_origin_e"])
        origin_n = float(self._metadata["grid_origin_n"])
        size = float(self._metadata["planner_cell_size_m"])
        c0 = int(np.floor((east_m - radius_m - origin_e) / size))
        c1 = int(np.floor((east_m + radius_m - origin_e) / size))
        r0 = int(np.floor((origin_n - (north_m + radius_m)) / size))
        r1 = int(np.floor((origin_n - (north_m - radius_m)) / size))
        return r0, r1, c0, c1

    def tile_at(self, east_m, north_m):
        r0, _, c0, _ = self._row_col_range(east_m, north_m, 0)
        row = self.connection.execute("SELECT * FROM tiles WHERE row=? AND col=?", (r0, c0)).fetchone()
        if row is None:
            return None
        columns = [description[0] for description in self.connection.execute("SELECT * FROM tiles LIMIT 1").description]
        return dict(zip(columns, row))

    def query_nearby(self, east_m, north_m, radius_m, max_landmarks):
        """Return landmarks inside a circular metric query, capped by response."""
        if max_landmarks < 1:
            raise ValueError("max_landmarks must be positive")
        r0, r1, c0, c1 = self._row_col_range(east_m, north_m, radius_m)
        tiles = self.connection.execute(
            "SELECT tile_id, quality FROM tiles WHERE row BETWEEN ? AND ? AND col BETWEEN ? AND ?",
            (r0, r1, c0, c1),
        ).fetchall()
        if not tiles:
            return _empty_batch()
        quality = {tile_id: value for tile_id, value in tiles}
        placeholders = ",".join("?" for _ in tiles)
        rows = self.connection.execute(f"""
            SELECT tile_id, east_m, north_m, up_m, pixel_col, pixel_row, scale, angle_deg,
                   response, octave, slope, normal_e, normal_n, normal_u, descriptor
            FROM landmarks WHERE tile_id IN ({placeholders}) ORDER BY response DESC
        """, tuple(quality)).fetchall()
        selected = [row for row in rows if (row[1] - east_m) ** 2 + (row[2] - north_m) ** 2 <= radius_m ** 2]
        selected = selected[:max_landmarks]
        if not selected:
            return _empty_batch()
        return LandmarkBatch(
            map_xyz=np.asarray([[row[1], row[2], row[3]] for row in selected], dtype=np.float32),
            descriptors=np.frombuffer(b"".join(row[14] for row in selected), dtype=np.uint8).reshape(len(selected), 32),
            keypoints=np.asarray([[row[4], row[5], row[6], row[7], row[8], row[9]] for row in selected], dtype=np.float32),
            terrain=np.asarray([[row[10], row[11], row[12], row[13]] for row in selected], dtype=np.float32),
            tile_quality=np.asarray([quality[row[0]] for row in selected], dtype=np.float32),
        )


def _empty_batch():
    return LandmarkBatch(np.empty((0, 3), np.float32), np.empty((0, 32), np.uint8),
                         np.empty((0, 6), np.float32), np.empty((0, 4), np.float32),
                         np.empty((0,), np.float32))
