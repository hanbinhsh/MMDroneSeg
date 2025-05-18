import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

class BlipCaptionGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    def generate_caption(self, image: Image.Image) -> str:
        image = image.convert('RGB')
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption

    def generate_captions_for_folder(self, image_folder: str, output_json_path: str):
        captions = {}
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]
        for img_file in tqdm(image_files, desc="Generating captions"):
            img_path = os.path.join(image_folder, img_file)
            try:
                image = Image.open(img_path)
                caption = self.generate_caption(image)
                captions[img_file] = caption
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                captions[img_file] = "An image"

        with open(output_json_path, 'w') as f:
            json.dump(captions, f, indent=4)
        print(f"\nCaptions saved to {output_json_path}")


if __name__ == '__main__':
    image_folder = '../dataset/Drone/classes_dataset/classes_dataset/original_images'
    output_json_path = 'clip_embeddings/captions_drone.json'

    generator = BlipCaptionGenerator()
    generator.generate_captions_for_folder(image_folder, output_json_path)
