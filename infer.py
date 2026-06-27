'''
Script to use ViTSTR to convert scene text image(s) to text.

Usage - single image:
    python infer.py --image path/to/image.png --model saved_models/.../best_accuracy.pth

Usage - whole folder:
    python infer.py --image_folder path/to/folder --model saved_models/.../best_accuracy.pth

Optional flags:
    --output results.txt   Save results to a text file (tab-separated: filename <TAB> prediction)
    --gpu                  Run on GPU
    --quantized            Use a quantized model
    --rpi                  Use qnnpack backend (Raspberry Pi / ARM)
'''

import sys
import os
import torch
import torch.nn as nn
import validators
import time
from infer_utils import TokenLabelConverter, NormalizePAD, ViTSTRFeatureExtractor
from infer_utils import get_args
from modules.vitstr import create_vitstr

# Persian character set
PERSIAN_CHARS = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءآ"

# Image extensions considered valid
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp'}


def collect_image_files(path):
    """
    Return a sorted list of image file paths.
    - If path is a file, return [path].
    - If path is a folder, return all image files inside it (non-recursive).
    """
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        files = sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])
        if not files:
            raise ValueError(f"No image files found in folder: {path}")
        return files
    else:
        raise ValueError(f"Path does not exist: {path}")


def img2text(model, images, converter):
    pred_strs = []
    with torch.no_grad():
        for img in images:
            pred = model(img, seqlen=converter.batch_max_length)
            _, pred_index = pred.topk(1, dim=-1, largest=True, sorted=True)
            pred_index = pred_index.view(-1, converter.batch_max_length)
            length_for_pred = torch.IntTensor([converter.batch_max_length - 1])
            pred_str = converter.decode(pred_index[:, 1:], length_for_pred)
            pred_EOS = pred_str[0].find('[s]')
            pred_str = pred_str[0][:pred_EOS]
            pred_strs.append(pred_str)
    return pred_strs


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    converter = TokenLabelConverter(args)
    args.num_class = len(converter.character)
    extractor = ViTSTRFeatureExtractor()

    # -- collect files ----------------------------------------------------------
    if args.time:
        # benchmark mode: use the fixed demo images
        files = [
            "demo_1.png", "demo_2.jpg", "demo_3.png", "demo_4.png", "demo_5.png",
            "demo_6.png", "demo_7.png", "demo_8.jpg", "demo_9.jpg", "demo_10.jpg"
        ]
        files = [os.path.join("demo_image", f) for f in files]
    else:
        # prefer --image_folder, fall back to --image
        input_path = args.image_folder if args.image_folder else args.image
        if input_path is None:
            raise ValueError("Provide --image (single file) or --image_folder (directory).")
        files = collect_image_files(input_path)
        print(f"Found {len(files)} image(s) to process.")

    # -- load images ------------------------------------------------------------
    images = []
    valid_files = []
    for f in files:
        try:
            img = extractor(f)
            if args.gpu:
                img = img.to(device)
            images.append(img)
            valid_files.append(f)
        except Exception as e:
            print(f"[Warning] Could not load {f}: {e} - skipping.")

    if not images:
        raise RuntimeError("No images could be loaded.")

    # -- load model -------------------------------------------------------------
    if args.quantized:
        backend = "qnnpack" if args.rpi else "fbgemm"
        torch.backends.quantized.engine = backend

    if validators.url(args.model):
        checkpoint_path = args.model.rsplit('/', 1)[-1]
        torch.hub.download_url_to_file(args.model, checkpoint_path)
    else:
        checkpoint_path = args.model

    if args.quantized:
        # JIT-compiled or quantized model: load directly as a full model object
        model = torch.jit.load(checkpoint_path, map_location=device)
    else:
        # best_accuracy.pth from train.py saves a DataParallel state dict,
        # so we must build the model first and then load the weights into it.
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Build the ViTSTR model
        # Use --TransformerModel if provided, otherwise default to vitstr_small_patch16_224
        transformer_model = getattr(args, 'TransformerModel', 'vitstr_small_patch16_224') or 'vitstr_small_patch16_224'
        print(f'Building model architecture: {transformer_model}')
        model = create_vitstr(num_tokens=args.num_class, model=transformer_model)

        # The state dict may have 'module.' prefix (from DataParallel) - strip it
        from collections import OrderedDict
        clean_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # strip 'module.' and 'module.vitstr.' prefixes
            if k.startswith('module.vitstr.'):
                clean_state_dict[k[len('module.vitstr.'):]] = v
            elif k.startswith('module.'):
                clean_state_dict[k[len('module.'):]] = v
            else:
                clean_state_dict[k] = v

        model.load_state_dict(clean_state_dict, strict=False)

    if args.gpu:
        model.to(device)

    model.eval()

    # -- benchmark timing mode --------------------------------------------------
    if args.time:
        n_times = 10
        n_total = len(images) * n_times
        [img2text(model, images, converter) for _ in range(n_times)]  # warm-up
        start_time = time.time()
        [img2text(model, images, converter) for _ in range(n_times)]
        end_time = time.time()
        ave_time = (end_time - start_time) / n_total
        print("Average inference time per image: %0.2e sec" % ave_time)

    # -- run inference ----------------------------------------------------------
    pred_strs = img2text(model, images, converter)

    return zip(valid_files, pred_strs)


if __name__ == '__main__':
    args = get_args()
    # Use Persian character set
    args.character = PERSIAN_CHARS

    results = list(infer(args))

    # -- print results ----------------------------------------------------------
    def safe_print(text):
        """Print text correctly in Jupyter, Windows terminal, or any environment."""
        try:
            # Works in normal terminal - writes raw UTF-8 bytes
            sys.stdout.buffer.write((text + '\n').encode('utf-8'))
            sys.stdout.buffer.flush()
        except AttributeError:
            # Jupyter replaces stdout with its own object (no .buffer);
            # plain print() works here since Jupyter handles Unicode natively.
            print(text)

    safe_print(f"\n{'-' * 60}")
    safe_print(f"{'Image':<40} Prediction")
    safe_print(f"{'-' * 60}")
    for filename, text in results:
        short_name = os.path.basename(filename)
        safe_print(f"{short_name:<40} {text}")
    safe_print(f"{'-' * 60}")

    # -- optionally save to file ------------------------------------------------
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for filename, text in results:
                f.write(f"{filename}\t{text}\n")
        print(f"\nResults saved to: {args.output}")