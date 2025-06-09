import os
import cv2
from pathlib import Path

def crop_images_with_expansion(
    image_dir, label_dir,
    output_image_dir, output_label_dir,
    num_rows, num_cols, f=0
):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_image_dir = Path(output_image_dir)
    output_label_dir = Path(output_label_dir)

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in image_dir.glob("*.png")])
    count = 0

    for img_path in image_files:
        label_path = label_dir / img_path.name
        if not label_path.exists():
            print(f"Label not found for {img_path.name}, skipping.")
            continue

        img = cv2.imread(str(img_path))
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

        h, w = img.shape[:2]
        crop_h = h // num_rows
        crop_w = w // num_cols

        for i in range(num_rows):
            if i == 0:
                y1 = 0
                y2 = crop_h + f
            elif i == num_rows - 1:
                y1 = h - crop_h - f
                y2 = h
            else:
                y1 = i * crop_h - f // 2
                y2 = (i + 1) * crop_h + f // 2
            y1 = max(0, y1)
            y2 = min(h, y2)

            for j in range(num_cols):
                if j == 0:
                    x1 = 0
                    x2 = crop_w + f
                elif j == num_cols - 1:
                    x1 = w - crop_w - f
                    x2 = w
                else:
                    x1 = j * crop_w - f // 2
                    x2 = (j + 1) * crop_w + f // 2
                x1 = max(0, x1)
                x2 = min(w, x2)

                cropped_img = img[y1:y2, x1:x2]
                cropped_label = label[y1:y2, x1:x2]

                filename = f"{count:05d}.png"
                cv2.imwrite(str(output_image_dir / filename), cropped_img)
                cv2.imwrite(str(output_label_dir / filename), cropped_label)

                # print(f"{filename}: size = {x2 - x1}x{y2 - y1}")
                count += 1

    print(f"\n All done! Total cropped image pairs: {count}")

if __name__ == '__main__':
    crop_images_with_expansion(
        image_dir='../dataset/Drone/classes_dataset/classes_dataset/val_original_pc',
        label_dir='../dataset/Drone/classes_dataset/classes_dataset/val_label_pc',
        output_image_dir='../dataset/Drone/classes_dataset/classes_dataset/val_original',
        output_label_dir='../dataset/Drone/classes_dataset/classes_dataset/val_label',
        num_rows=3,
        num_cols=3,
        f=16
    )
