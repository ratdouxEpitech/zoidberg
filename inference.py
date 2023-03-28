import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os


def preprocess_image(image_path, img_height, img_width):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # expand dimensions for batch input
    return img_array


def main():
    normal_folder = "chest_Xray\\test\\NORMAL"
    pneumonia_folder = "chest_Xray\\test\\PNEUMONIA"
    models_folder = r"models"
    img_height = 256
    img_width = 256

    model_results = []

    for model_name in os.listdir(models_folder):
        model_path = os.path.join(models_folder, model_name)
        if not os.path.isfile(model_path):
            continue

        loaded_model = load_model(model_path)

        def is_prediction_correct(image_path, true_class):
            preprocessed_image = preprocess_image(image_path, img_height, img_width)
            prediction = loaded_model.predict(preprocessed_image)
            predicted_class = "pneumonia" if prediction[0][0] > 0.5 else "normal"
            return predicted_class == true_class

        results = {
            "normal": {"correct": 0, "total": 0},
            "pneumonia": {"correct": 0, "total": 0},
        }

        for folder, true_class in [(normal_folder, "normal"), (pneumonia_folder, "pneumonia")]:
            for image_name in os.listdir(folder):
                image_path = os.path.join(folder, image_name)
                results[true_class]["total"] += 1
                if is_prediction_correct(image_path, true_class):
                    results[true_class]["correct"] += 1

        total_correct = results["normal"]["correct"] + results["pneumonia"]["correct"]
        total_images = results["normal"]["total"] + results["pneumonia"]["total"]
        normal_correct_pct = (results["normal"]["correct"] / results["normal"]["total"]) * 100
        pneumonia_correct_pct = (results["pneumonia"]["correct"] / results["pneumonia"]["total"]) * 100
        overall_correct_pct = (total_correct / total_images) * 100

        model_results.append({
            "name": model_name,
            "results": results,
            "total_correct": total_correct,
            "normal_correct_pct": normal_correct_pct,
            "pneumonia_correct_pct": pneumonia_correct_pct,
            "overall_correct_pct": overall_correct_pct
        })

    sorted_by_total_correct = sorted(model_results, key=lambda x: x["total_correct"], reverse=True)
    sorted_by_normal_correct_pct = sorted(model_results, key=lambda x: x["normal_correct_pct"], reverse=True)
    sorted_by_pneumonia_correct_pct = sorted(model_results, key=lambda x: x["pneumonia_correct_pct"], reverse=True)

    with open("detailed_results.txt", "w") as f:
        for model_result in model_results:
            f.write(f"Model: {model_result['name']}\n\n")
            f.write("Results:\n")
            f.write(f"Total Images: {total_images}\n")
            f.write(f"  Tcorrect: {model_result['total_correct']} ({model_result['overall_correct_pct']:.2f}%)\n")
            f.write(f"  Tincorrect: {total_images - model_result['total_correct']}\n")
            f.write(f"Normal:\n")
            f.write(f"  Ncorrect: {model_result['results']['normal']['correct']} ({model_result['normal_correct_pct']:.2f}%)\n")
            f.write(f"  Nincorrect: {model_result['results']['normal']['total'] - model_result['results']['normal']['correct']}\n")
            f.write(f"  Nimages: {model_result['results']['normal']['total']}\n")
            f.write("\n")
            f.write(f"Pneumonia:\n")
            f.write(f"  Pcorrect: {model_result['results']['pneumonia']['correct']} ({model_result['pneumonia_correct_pct']:.2f}%)\n")
            f.write(f"  Pincorrect: {model_result['results']['pneumonia']['total'] - model_result['results']['pneumonia']['correct']}\n")
            f.write(f"  Pimages: {model_result['results']['pneumonia']['total']}\n")
            f.write("\n")
            f.write("=" * 80 + "\n\n")

        f.write("Rankings:\n\n")

        f.write("By Total Correct Hits:\n")
        for rank, model_result in enumerate(sorted_by_total_correct, start=1):
            f.write(f"{rank}. {model_result['name']} - {model_result['total_correct']} ({model_result['overall_correct_pct']:.2f}%)\n")

        f.write("\n")

        f.write("By Normal Correct Percentage:\n")
        for rank, model_result in enumerate(sorted_by_normal_correct_pct, start=1):
            f.write(f"{rank}. {model_result['name']} - {model_result['results']['normal']['correct']} ({model_result['normal_correct_pct']:.2f}%)\n")

        f.write("\n")

        f.write("By Pneumonia Correct Percentage:\n")
        for rank, model_result in enumerate(sorted_by_pneumonia_correct_pct, start=1):
            f.write(f"{rank}. {model_result['name']} - {model_result['results']['pneumonia']['correct']} ({model_result['pneumonia_correct_pct']:.2f}%)\n")

    print("Detailed results and rankings written to 'detailed_results.txt'")

if __name__ == "__main__":
    main()
