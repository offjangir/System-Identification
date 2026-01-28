import numpy as np
import sys

def main(path):
    data = np.load(path)

    print("=" * 60)
    print(f"Loaded: {path}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print("=" * 60)

    # Basic stats
    print("Min:", data.min(axis=0))
    print("Max:", data.max(axis=0))
    print("Mean:", data.mean(axis=0))
    print("Std:", data.std(axis=0))

    print("\nFirst 5 timesteps:")
    print(data[:5])

    print("\nLast 5 timesteps:")
    print(data[-5:])

    # Heuristic interpretation
    dim = data.shape[1]
    print("\n--- Interpretation ---")

    if dim == 3:
        print("→ Likely XYZ position only (meters)")
    elif dim == 6:
        print("→ Likely [x, y, z, roll, pitch, yaw]")
    elif dim == 7:
        print("→ Likely [x, y, z, qx, qy, qz, qw]")
    else:
        print("→ Unknown format")

    # Check scale
    max_abs = np.abs(data[:, :3]).max()
    if max_abs < 5:
        print("✓ Position scale looks like meters")
    else:
        print("⚠ Position scale unusually large — check units")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_cartesian.py path/to/episode_xxx_cartesian_position.npy")
        sys.exit(1)
    main(sys.argv[1])
