"""Inspect a landmark SQLite package and optionally execute a bounded query."""

import argparse
import json

from .database import LandmarkDatabase


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect an onboard landmark database")
    parser.add_argument("--database", required=True)
    parser.add_argument("--east-m", type=float)
    parser.add_argument("--north-m", type=float)
    parser.add_argument("--radius-m", type=float, default=100.0)
    parser.add_argument("--max-landmarks", type=int, default=300)
    args = parser.parse_args(argv)
    with LandmarkDatabase(args.database) as database:
        metadata = database.metadata()
        tile_count = database.connection.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        landmark_count = database.connection.execute("SELECT COUNT(*) FROM landmarks").fetchone()[0]
        print(json.dumps({"metadata": metadata, "tile_count": tile_count,
                          "landmark_count": landmark_count}, indent=2, sort_keys=True))
        if (args.east_m is None) != (args.north_m is None):
            parser.error("--east-m and --north-m must be supplied together")
        if args.east_m is not None:
            batch = database.query_nearby(args.east_m, args.north_m, args.radius_m, args.max_landmarks)
            print(json.dumps({"query_landmarks": int(len(batch.map_xyz)),
                              "descriptor_shape": list(batch.descriptors.shape)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
