from flask import Flask, request, render_template
import pandas as pd
from urllib.request import urlopen
from PIL import Image
import timm
import torch

app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('index.html')  # Renders the homepage


# Load class mappings
def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        return {i: line.strip() for i, line in enumerate(f)}

# Load species mappings
def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    return df.set_index('species_id')['species'].to_dict()

# Load the model
def initialize_model(pretrained_path, num_classes, device):
    model = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=False,
        num_classes=num_classes,
        checkpoint_path=pretrained_path
    )
    model = model.to(device)
    return model.eval()

# Predict species from image
def predict_species(image_path, model, transforms, cid_to_spid, spid_to_sp, device):
    img = Image.open(urlopen(image_path) if 'http' in image_path else image_path)
    img = img.convert('RGB')  # Ensure the image is in RGB format
    img = transforms(img).unsqueeze(0).to(device)
    output = model(img)
    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
    top5_probabilities = top5_probabilities.cpu().detach().numpy()
    top5_class_indices = top5_class_indices.cpu().detach().numpy()

    predictions = []
    for proba, cid in zip(top5_probabilities[0], top5_class_indices[0]):
        species_id = cid_to_spid[cid]
        species = spid_to_sp[species_id]
        predictions.append((species_id, species, proba))
    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    image_path = file

    # Initialize mappings and model
    class_mapping_file = 'class_mapping.txt'
    species_mapping_file = 'species_id_to_name.txt'
    pretrained_path = 'model_best.pth.tar'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cid_to_spid = load_class_mapping(class_mapping_file)
    spid_to_sp = load_species_mapping(species_mapping_file)
    model = initialize_model(pretrained_path, len(cid_to_spid), device)

    # Get model-specific transforms
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Get predictions
    predictions = predict_species(image_path, model, transforms, cid_to_spid, spid_to_sp, device)
    
    results = [
        {
            "species_id": species_id,
            "species": species,
            "probability": float(proba)  # Convert to Python float
        }
        for species_id, species, proba in predictions
    ]

    # Render the results in predict.html
    return render_template('predict.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
