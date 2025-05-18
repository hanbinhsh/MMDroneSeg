import os
import torch
import json
from tqdm import tqdm
import clip


def extract_clip_text_embeddings(captions_json_path, output_path, device):
    # 加载 CLIP 模型
    clip_model, _ = clip.load("ViT-B/32", device=device)

    # 加载 captions.json
    try:
        with open(captions_json_path, "r") as f:
            captions = json.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到 {captions_json_path}")
        return

    text_embeddings = {}

    # 遍历所有图像文件名和对应 caption
    for filename, caption in tqdm(captions.items(), desc="生成CLIP文本嵌入"):
        text_token = clip.tokenize([caption]).to(device)
        with torch.no_grad():
            text_feat = clip_model.encode_text(text_token)
        text_embeddings[filename] = text_feat.cpu().numpy().tolist()

    # 保存嵌入向量
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(text_embeddings, f, indent=4)
    print(f"已保存文本嵌入至 {output_path}")


def main():
    captions_path = "clip_embeddings/captions_drone.json"
    output_path = "clip_embeddings/clip_text_embeddings_drone.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extract_clip_text_embeddings(captions_path, output_path, device)


if __name__ == "__main__":
    main()
