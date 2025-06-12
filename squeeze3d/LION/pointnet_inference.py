import os
import sys
import torch
import numpy as np
import argparse
from plyfile import PlyData, PlyElement
import importlib

# Add path to the models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(
    os.path.join(
        os.path.dirname(__file__), "../", "Pointnet_Pointnet2_pytorch", "models"
    )
)


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("PointNet2 Inference on PLY files")
    parser.add_argument("--ply_file", required=True, help="Path to the PLY file")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--model",
        default="pointnet2_cls_ssg",
        help="Model name [default: pointnet2_cls_ssg]",
    )
    parser.add_argument(
        "--num_point", type=int, default=1024, help="Point Number [default: 1024]"
    )
    parser.add_argument(
        "--num_class",
        type=int,
        default=40,
        choices=[10, 40],
        help="Number of classes [default: 40]",
    )
    parser.add_argument(
        "--use_normals", action="store_true", default=False, help="Use normals"
    )
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="Use CPU instead of GPU"
    )
    parser.add_argument(
        "--gpu", type=str, default="0", help="GPU to use [default: GPU 0]"
    )
    return parser.parse_args()


def load_ply(file_path, use_normals=True):
    """
    Load PLY file and convert to numpy array of points (and normals if available)
    """
    try:
        plydata = PlyData.read(file_path)

        # Get the vertex element from the PLY data
        vertex = plydata["vertex"]

        # Extract x, y, z coordinates
        x = vertex["x"]
        y = vertex["y"]
        z = vertex["z"]

        # Stack into a point cloud array
        pc = np.column_stack((x, y, z))

        # Check if normals are available in the PLY file
        has_normals = False
        try:
            if all(n in vertex.data.dtype.names for n in ["nx", "ny", "nz"]):
                has_normals = True
        except (AttributeError, TypeError):
            # Handle case where dtype.names is not accessible
            try:
                # Try direct access to see if fields exist
                nx_test = vertex["nx"]
                ny_test = vertex["ny"]
                nz_test = vertex["nz"]
                has_normals = True
            except (ValueError, KeyError):
                has_normals = False

        if use_normals and has_normals:
            # Extract and stack normal vectors
            try:
                nx = vertex["nx"]
                ny = vertex["ny"]
                nz = vertex["nz"]
                normals = np.column_stack((nx, ny, nz))
                # Concatenate points and normals
                pc = np.concatenate([pc, normals], axis=1)
            except (ValueError, KeyError) as e:
                print(f"Warning: Error accessing normals: {e}")
                normals = np.zeros_like(pc)
                pc = np.concatenate([pc, normals], axis=1)
        elif use_normals:
            print(
                "Warning: Normals requested but not found in PLY file. Using zeros instead."
            )
            normals = np.zeros_like(pc)
            pc = np.concatenate([pc, normals], axis=1)

        return pc
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def pc_normalize(pc):
    """
    Normalize point cloud to unit sphere
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def process_point_cloud(point_cloud, num_point=1024, use_normals=True):
    """
    Process the point cloud for inference
    """
    # Make sure we have the right number of points
    if point_cloud.shape[0] >= num_point:
        # Randomly sample points
        idx = np.random.choice(point_cloud.shape[0], num_point, replace=False)
    else:
        # If too few points, repeat some
        idx = np.random.choice(point_cloud.shape[0], num_point, replace=True)

    sampled_pc = point_cloud[idx, :]

    # Normalize the point cloud coordinates (xyz only)
    sampled_pc[:, 0:3] = pc_normalize(sampled_pc[:, 0:3])

    # If we don't want normals, only keep xyz
    if not use_normals:
        sampled_pc = sampled_pc[:, 0:3]

    # Convert to PyTorch tensor and correct shape for model input
    # PointNet2 expects [batch_size, channels, num_points]
    pc_tensor = torch.from_numpy(sampled_pc).float().unsqueeze(0)
    pc_tensor = pc_tensor.transpose(2, 1)

    return pc_tensor


def get_model_and_labels(model_name, num_class, normal_channel=True):
    """
    Load model and class labels
    """
    # Import the model module
    MODEL = importlib.import_module(model_name)

    # Get the model with specified number of classes
    classifier = MODEL.get_model(num_class=num_class, normal_channel=normal_channel)

    # Define labels for ModelNet40
    if num_class == 40:
        LABEL_MAP = {
            0: "airplane",
            1: "bathtub",
            2: "bed",
            3: "bench",
            4: "bookshelf",
            5: "bottle",
            6: "bowl",
            7: "car",
            8: "chair",
            9: "cone",
            10: "cup",
            11: "curtain",
            12: "desk",
            13: "door",
            14: "dresser",
            15: "flower_pot",
            16: "glass_box",
            17: "guitar",
            18: "keyboard",
            19: "lamp",
            20: "laptop",
            21: "mantel",
            22: "monitor",
            23: "night_stand",
            24: "person",
            25: "piano",
            26: "plant",
            27: "radio",
            28: "range_hood",
            29: "sink",
            30: "sofa",
            31: "stairs",
            32: "stool",
            33: "table",
            34: "tent",
            35: "toilet",
            36: "tv_stand",
            37: "vase",
            38: "wardrobe",
            39: "xbox",
        }
    # Define labels for ModelNet10
    elif num_class == 10:
        LABEL_MAP = {
            0: "bathtub",
            1: "bed",
            2: "chair",
            3: "desk",
            4: "dresser",
            5: "monitor",
            6: "night_stand",
            7: "sofa",
            8: "table",
            9: "toilet",
        }
    else:
        print(f"Error: Unsupported number of classes: {num_class}")
        sys.exit(1)

    return classifier, LABEL_MAP


def main():
    """
    Main function for inference
    """
    args = parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Check if files exist
    if not os.path.exists(args.ply_file):
        print(f"Error: PLY file {args.ply_file} not found")
        return

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found")
        return

    # Set normal usage flag
    use_normals = args.use_normals

    print(f"Configuration:")
    print(f" - Model: {args.model}")
    print(f" - Using normals: {'Yes' if args.use_normals else 'No'}")
    print(f" - Number of points: {args.num_point}")
    print(f" - Number of classes: {args.num_class}")
    print(f" - Device: {'CPU' if args.use_cpu else f'GPU:{args.gpu}'}")

    try:
        print(f"\nLoading PLY file: {args.ply_file}")
        # Load the point cloud
        point_cloud = load_ply(args.ply_file, use_normals=use_normals)
        print(
            f"Loaded point cloud with {point_cloud.shape[0]} points and {point_cloud.shape[1]} features"
        )

        # Check if we have the right number of features
        expected_features = 6 if use_normals else 3
        if point_cloud.shape[1] != expected_features:
            print(
                f"Warning: Expected {expected_features} features per point but got {point_cloud.shape[1]}"
            )
            if point_cloud.shape[1] == 3 and use_normals:
                print(
                    "The PLY file doesn't contain normal data, but you're trying to use a model with normals."
                )
                print("Consider either:")
                print(" 1. Using a PLY file that includes normal data")
                print(
                    " 2. Remove the --use_normals flag if using a model without normal features"
                )
                return

        # Process the point cloud for model input
        processed_pc = process_point_cloud(
            point_cloud, args.num_point, use_normals=use_normals
        )

        # Load the model and class labels
        print(f"\nLoading model: {args.model}")
        classifier, label_map = get_model_and_labels(
            args.model, args.num_class, normal_channel=use_normals
        )

        # Load the pre-trained model weights
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu"
        )  # Load to CPU first
        classifier.load_state_dict(checkpoint["model_state_dict"])

        # Set the model to evaluation mode
        classifier.eval()

        # Move model and data to GPU if available
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not args.use_cpu else "cpu"
        )
        print(f"Using device: {device}")
        classifier = classifier.to(device)
        processed_pc = processed_pc.to(device)

        # Perform inference
        print("Performing inference...")
        with torch.no_grad():
            pred, _ = classifier(processed_pc)
            pred_choice = pred.data.max(1)[1]
            predicted_class = pred_choice.item()

            # Get prediction probabilities
            probabilities = torch.nn.functional.softmax(pred, dim=1)[0]
            top_5_probs, top_5_indices = torch.topk(
                probabilities, min(5, args.num_class)
            )

        # Print results
        print("\n" + "=" * 50)
        print(
            f"Predicted class: {label_map[predicted_class]} (Class ID: {predicted_class})"
        )
        print("=" * 50)

        print("\nTop 5 predictions:")
        print("-" * 50)
        print("Rank | Class ID | Class Name              | Probability")
        print("-" * 50)
        for i in range(len(top_5_indices)):
            idx = top_5_indices[i].item()
            prob = top_5_probs[i].item() * 100
            print(f"{i+1:4d} | {idx:8d} | {label_map[idx]:<22s} | {prob:.2f}%")

        print("\nInference completed successfully!")

    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback

        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure the PLY file is properly formatted")
        print("2. Ensure the checkpoint is compatible with the selected model")
        print("3. Check that the use of normals matches how the model was trained")


if __name__ == "__main__":
    main()
