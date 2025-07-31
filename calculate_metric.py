import argparse
import os.path as osp
from dataset_utils import DetValidator, OBBValidator, SegmentValidator


def parse_names_from_file(file_path):
    """
    从文件中解析类别名称，每行一个。
    返回一个字典，其中键是行号（从0开始），值是类别名称。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 使用 enumerate 获取行号（索引从0开始）
            # line.strip() 用于移除每行前后可能存在的空白字符（包括换行符）
            names = {i: line.strip() for i, line in enumerate(f) if line.strip()}
        
        if not names:
            raise ValueError("The names file is empty or contains no valid class names.")
            
        return names
    except FileNotFoundError:
        # 使用 argparse.ArgumentTypeError 可以让 argparse 优雅地处理错误并显示给用户
        raise argparse.ArgumentTypeError(f"The names file was not found at: {file_path}")
    except Exception as e:
        raise argparse.ArgumentTypeError(f"An error occurred while reading the names file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Calculate detection/segmentation metrics from model predictions.")
    subparsers = parser.add_subparsers(dest="task", required=True, help="Specify the validation task: obb, hbb, or seg.")

    # 定义通用的参数，避免重复
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--pred_path", type=str, required=True, help="Path to the prediction label files directory.")
    common_parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth label files directory.")
    common_parser.add_argument("--val_list_path", type=str, required=True, help="Path to the validation list file (e.g., val.txt).")
    common_parser.add_argument("--names_file", type=str, required=True, help="Path to a text file containing class names, one per line.")

    # 使用 parents 参数来继承通用参数
    subparsers.add_parser("obb", help="Oriented Bounding Box (OBB) validation.", parents=[common_parser])
    subparsers.add_parser("hbb", help="Horizontal Bounding Box (HBB) validation.", parents=[common_parser])
    subparsers.add_parser("seg", help="Instance Segmentation validation.", parents=[common_parser])

    args = parser.parse_args()

    # 从文件解析 names
    try:
        class_names = parse_names_from_file(args.names_file)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e)) # argparse 会处理这个错误并退出

    print(f"Running task: {args.task}")
    print(f"Prediction Path: {args.pred_path}")
    print(f"Ground Truth Path: {args.gt_path}")
    print(f"Validation List: {args.val_list_path}")
    print(f"Class Names File: {args.names_file}")
    print(f"Parsed Class Names ({len(class_names)} total): {class_names}")

    if args.task == "obb":
        validator = OBBValidator(
            pred_path=args.pred_path,
            gt_path=args.gt_path,
            val_list_path=args.val_list_path,
            names=class_names
        )
    elif args.task == "hbb":
        validator = DetValidator(
            pred_path=args.pred_path,
            gt_path=args.gt_path,
            val_list_path=args.val_list_path,
            names=class_names
        )
    elif args.task == "seg":
        validator = SegmentValidator(
            pred_path=args.pred_path,
            gt_path=args.gt_path,
            val_list_path=args.val_list_path,
            names=class_names
        )
    else:
        print(f"Unknown task: {args.task}")
        return

    validator.cal_metrics()
    print(f"\nMetrics calculation for task '{args.task}' completed successfully.")


if __name__ == "__main__":
    main()