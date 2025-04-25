import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import DistilBertTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings

from utils import load_config, ImageEncoder, TextEncoder, SigLIP


def plot_results(image_path, pil_image, candidate_labels, probs):
    """
    Displays the image and classification probabilities.

    Args:
        image_path (str): Path to the input image.
        pil_image (PIL.Image): PIL image object.
        candidate_labels (list[str]): List of candidate labels.
        probs (list[float]): Probabilities corresponding to the labels.
    """
    plt.figure(figsize=(7, 7))
    plt.imshow(pil_image)
    plt.axis('off')

    sorted_pairs = sorted(zip(candidate_labels, probs), key=lambda x: x[1], reverse=True)
    if not sorted_pairs:
        plt.title("No predictions available", fontsize=12)
        prob_text = "N/A"
    else:
        best_label, best_prob = sorted_pairs[0]
        plt.title(f"Predicted: {best_label} ({best_prob:.1%})", fontsize=12)
        prob_text = "\n".join([f"{lbl}: {p:.1%}" for lbl, p in sorted_pairs])

    plt.text(0, pil_image.height * 1.05, prob_text, fontsize=9, verticalalignment='top')
    plt.tight_layout()
    plt.show()


def get_prompt_template_for_language(language: str) -> str:
    """
    Returns a prompt template string based on the specified language.

    Args:
        language (str): The desired language ('english', 'hindi', 'marathi', 'hinglish').

    Returns:
        str: A format string for the prompt template.
    """
    lang = language.lower() if language else 'english'

    if lang == 'hindi':
        return "यह {} का फोटो है"
    elif lang == 'marathi':
        return "{} चे छायाचित्र"
    elif lang == 'hinglish':
        return "ek photo {} ka"
    elif lang == 'english':
        return "a photo of a {}"
    else:
        warnings.warn(f"Unsupported language '{language}'. Defaulting to English template.", UserWarning)
        return "a photo of a {}"


def zero_shot_classifier(config, image_path, candidate_labels, target_language, model_path_override=None):
    """
    Performs zero-shot image classification using a trained SigLIP model.

    Args:
        config (SimpleNamespace): Loaded configuration object.
        image_path (str): Path to the input image file.
        candidate_labels (list[str]): List of possible labels for the image.
        target_language (str): Language for prompt generation ('english', 'hindi', 'marathi', 'hinglish').
        model_path_override (str, optional): Path to specific model weights (.pth). Defaults to None.

    Returns:
        dict: Dictionary of candidate labels and their probabilities.
    """
    print(f"--- Starting Zero-Shot Classification (Language: {target_language or 'Default English'}) ---")
    print(f"Image: {image_path}")
    print(f"Candidate Labels: {candidate_labels}")

    model_path = model_path_override if model_path_override else config.model_save_path
    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found at '{model_path}'")
        return None
    if not os.path.exists(image_path):
        print(f"ERROR: Input image not found at '{image_path}'")
        return None

    device = config.device if hasattr(config, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        image_encoder = ImageEncoder(config).to(device)
        text_encoder = TextEncoder(config).to(device)
        model = SigLIP(image_encoder, text_encoder, config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"ERROR initializing/loading model: {e}")
        return None

    try:
        img = Image.open(image_path).convert("RGB")
        img_transform = transforms.Compose([
            transforms.Resize(config.dataset.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = img_transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"ERROR processing image '{image_path}': {e}")
        return None

    prompt_template = get_prompt_template_for_language(target_language)
    prompts = [prompt_template.format(label) for label in candidate_labels]

    if not prompts:
        print("ERROR: No prompts were generated.")
        return None

    try:
        tokenizer = DistilBertTokenizer.from_pretrained(config.model.text_encoder.preset)
        inputs = tokenizer(
            prompts,
            max_length=config.dataset.text_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
    except Exception as e:
        print(f"ERROR tokenizing prompts: {e}")
        return None

    with torch.no_grad():
        image_features = model.image_encoder(image_tensor)
        text_features = model.text_encoder(inputs["input_ids"], inputs["attention_mask"])

        image_features = nn.functional.normalize(image_features, p=2, dim=-1)
        text_features = nn.functional.normalize(text_features, p=2, dim=-1)

        logits_per_image = image_features @ text_features.T
        logits_per_image = model.logit_scale * logits_per_image + model.logit_bias
        probs = torch.sigmoid(logits_per_image).cpu().numpy().flatten()

    results_dict = {label: float(prob) for label, prob in zip(candidate_labels, probs)}
    sorted_results = sorted(results_dict.items(), key=lambda item: item[1], reverse=True)

    print("\n--- Zero-Shot Classification Results ---")
    for label, prob in sorted_results:
        print(f"{label:>20s} : {prob:.4f} ({prob:.1%})")

    try:
        plot_results(image_path, img, candidate_labels, probs)
    except Exception as plot_err:
        print(f"Warning: Could not display plot. Error: {plot_err}")

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot image classification using SigLIP with language-specific prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--labels", nargs='+', required=True, help="Space-separated candidate labels.")
    parser.add_argument("--model_path", type=str, default=None, help="Override model path from config.")
    parser.add_argument(
        "--language", type=str, default='english',
        choices=['english', 'hindi', 'marathi', 'hinglish'],
        help="Language to use for generating text prompts."
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"FATAL Error loading config '{args.config}': {e}")
        exit(1)

    try:
        predictions = zero_shot_classifier(
            config, args.image, args.labels, args.language, args.model_path
        )
        if predictions:
            print("\nPrediction Dictionary:")
            print(predictions)
    except Exception as e:
        print(f"\nFATAL INFERENCE ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
