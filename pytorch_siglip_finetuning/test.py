import os
import argparse
import torch
from transformers import AutoProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib.font_manager import FontProperties
from utils import load_config, SigLIPWrapper

font_prop = FontProperties(fname='mangal.ttf', size=18)

def plot_results(image_path, pil_image, candidate_labels, probs, language=None):
    """
    Displays the image and classification probabilities.

    Args:
        image_path (str): Path to the input image.
        pil_image (PIL.Image.Image): PIL image object.
        candidate_labels (list): List of candidate labels.
        probs (list): List of probabilities corresponding to the labels.
        language (str, optional): Language for the text. Defaults to None.
    """
    plt.figure(figsize=(7, 7))
    plt.imshow(pil_image)
    plt.axis('off')
    
    use_font_prop = language is not None and language.lower() != "english"
    font_args = {'fontproperties': font_prop} if use_font_prop else {}

    sorted_pairs = sorted(zip(candidate_labels, probs), key=lambda x: x[1], reverse=True)
    
    if not sorted_pairs:
        plt.title("No predictions", fontsize=12, **font_args)
    else:
        best_label, best_prob = sorted_pairs[0]
        plt.title(f"Predicted: {best_label} ({best_prob:.1%})", fontsize=12, **font_args)

    plt.show()

def get_prompt_template_for_language(language):
    """
    Returns the prompt template for the specified language.

    Args:
        language (str): Language for the prompt.

    Returns:
        str: Prompt template.
    """
    lang = language.lower() if language else 'english'
    if lang == 'hindi': return "यह {} का फोटो है"
    elif lang == 'marathi': return "{} चे छायाचित्र"
    elif lang == 'hinglish': return "ek photo {} ka"
    elif lang == 'english': return "a photo of a {}"
    else:
        warnings.warn(f"Unsupported language '{language}'. Defaulting to English.", UserWarning)
        return "a photo of a {}"

def zero_shot_classifier(config, image_path, candidate_labels, target_language, model_path_override=None):
    """
    Performs zero-shot classification using a fine-tuned Hugging Face SigLIP model.

    Args:
        config (object): Configuration object with model and device details.
        image_path (str): Path to the input image.
        candidate_labels (list): List of candidate labels.
        target_language (str): Language for the prompts.
        model_path_override (str, optional): Override path for the fine-tuned model. Defaults to None.

    Returns:
        dict: Dictionary of labels and their probabilities.
    """
    print(f"--- Starting Zero-Shot Classification (Lang: {target_language or 'Default English'}) ---")
    print(f"Image: {image_path}")
    print(f"Labels: {candidate_labels}")

    model_path = model_path_override if model_path_override else config.model_save_path
    hf_model_name = config.model_name

    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found: '{model_path}'")
        return None
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: '{image_path}'")
        return None

    device = config.device if hasattr(config, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        processor = AutoProcessor.from_pretrained(hf_model_name)
    except Exception as e:
        print(f"ERROR loading processor '{hf_model_name}': {e}")
        return None

    try:
        model = SigLIPWrapper(hf_model_name, config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"ERROR initializing/loading model: {e}")
        return None

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"ERROR processing image '{image_path}': {e}")
        return None

    prompt_template = get_prompt_template_for_language(target_language)
    prompts = [prompt_template.format(label) for label in candidate_labels]

    if not prompts:
        print("ERROR: No prompts generated.")
        return None

    try:
        inputs = processor(
            text=prompts,
            images=img,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config.dataset.text_sequence_length
        ).to(device)
    except Exception as e:
        print(f"ERROR during HF processing: {e}")
        return None

    with torch.no_grad():
        try:
            logits_per_image = model(inputs)
            probs = torch.sigmoid(logits_per_image).cpu().numpy().flatten()
        except Exception as e:
            print(f"ERROR during model inference: {e}")
            return None

    results_dict = {label: float(prob) for label, prob in zip(candidate_labels, probs)}
    sorted_results = sorted(results_dict.items(), key=lambda item: item[1], reverse=True)

    print("\n--- Zero-Shot Classification Results ---")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prompt Language: {target_language or 'English (Default)'}")
    print("-" * 35)
    for label, prob in sorted_results:
        print(f"{label:>20s} : {prob:.4f} ({prob:.1%})")
    print("-" * 35)

    try:
        plot_results(image_path, img, candidate_labels, probs, language=target_language)
    except Exception as plot_err:
        print(f"Warning: Could not display plot: {plot_err}")

    print("--- Classification Finished ---")
    return results_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot classification with fine-tuned HF SigLIP.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--labels", nargs='+', required=True, help="Space-separated candidate labels.")
    parser.add_argument("--model_path", type=str, default=None, help="Override fine-tuned model path.")
    parser.add_argument("--language", type=str, default='english', choices=['english', 'hindi', 'marathi', 'hinglish'], help="Prompt language.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"FATAL Error loading config '{args.config}': {e}")
        exit(1)

    try:
        predictions = zero_shot_classifier(config, args.image, args.labels, args.language, args.model_path)
        if predictions:
            print("\nPrediction Dictionary:\n", predictions)
    except Exception as e:
        print(f"\nFATAL INFERENCE ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)