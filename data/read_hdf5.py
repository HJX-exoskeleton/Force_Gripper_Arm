import argparse
from pathlib import Path

import h5py


def describe(node, prefix=""):
    if isinstance(node, h5py.Dataset):
        print(f"{prefix}{node.name} shape={node.shape} dtype={node.dtype}")
        return

    print(f"{prefix}{node.name}/")
    for key in node.keys():
        describe(node[key], prefix + "  ")


def main():
    parser = argparse.ArgumentParser(description="Inspect one tactile imitation-learning HDF5 episode.")
    parser.add_argument(
        "path",
        nargs="?",
        default="/home/hjx/hjx_file/STF/Force_Gripper_Arm/mujoco/tactile_gripper/demo_gripper_2f85/runs/il_2f85/dataset/episode_00000.hdf5",
        help="Path to an episode HDF5 file.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with h5py.File(path, "r") as obj:
        print("attrs:")
        for key, value in obj.attrs.items():
            print(f"  {key}: {value}")
        print("structure:")
        describe(obj)


if __name__ == "__main__":
    main()
