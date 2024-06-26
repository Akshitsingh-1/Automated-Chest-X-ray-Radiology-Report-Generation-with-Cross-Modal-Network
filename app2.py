from flask import Flask, jsonify, redirect, url_for, request, render_template
from PIL import Image
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
import time
import numpy as np
from main_test import *
from user_keywords import *
from search import *
from full_report_translate import *

app = Flask(__name__)

# Preprocess Image
def preprocess_image(path1, path2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485), (0.229))
    ])
    image_1 = Image.open(path1).convert("RGB")  # (512, 420)
    image_2 = Image.open(path2).convert("RGB")  # (512, 420)
    image_1 = transform(image_1)  # [3, 224, 224]
    image_2 = transform(image_2)  # [3, 224, 224]
    image = torch.stack((image_1, image_2), 0)  # [2, 3, 224, 224]
    return image

# Make Prediction
def make_prediction(path1, path2, model, device):
    image = preprocess_image(path1, path2).to(device)  # Move image to the correct device
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0), mode='sample')  # Add batch dimension and move image to device
        report = model.tokenizer.decode_batch(output.cpu().numpy())
    return report

# Load your trained model
# Parse arguments
args = parse_agrs()

# Fix random seeds
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# Create tokenizer and model
tokenizer = Tokenizer(args)
model = R2GenModel(args, tokenizer)
load_path = "results/iu_xray/model_best.pth"
checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

# Move model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


@app.route('/')
def index():
    # Main page
    print("helloooo")
    return render_template('home.html')

# Move from home page to doctor page:
@app.route('/getStart', methods=['GET', 'POST'])
def getStart():
    if request.method == 'POST':
        print("starttttt")
        return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        seconds1 = time.time()
        # Get the file from post request
        f = []
        for item in request.files.getlist('file1'):
            f.append(item)
        f1 = request.files['file1']
        f2 = request.files['file1']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path1 = os.path.join(
            basepath, 'uploads', secure_filename(f[0].filename))
        file_path2 = os.path.join(
            basepath, 'uploads', secure_filename(f[1].filename))
        f[0].save(file_path1)
        f[1].save(file_path2)

        # Make prediction
        pred_report = make_prediction(file_path1, file_path2, model, device)

        # result = str(pred_class[0][0][1])               # Convert to string
        seconds2 = time.time()
        print("totalTime: ", seconds2-seconds1)
        importantKeywords = getkewords(pred_report[0])
        res = pred_report[0]  # + "\n\n Important Keywords: "
        keys = ""
        source_lang = 'en'
        target_lang = 'hi'

        for word in importantKeywords:
            keys += word + " : "
            all_possible_meaing = understand_words(
                word, source_lang, target_lang)
            all_possible_meaing = all_possible_meaing[:3]
            del all_possible_meaing[0]
            L = len(all_possible_meaing)
            count = 0
            for term in all_possible_meaing:
                keys += term
                count += 1
                if count <= L-1:
                    keys += " - "
            keys += " <br>    "
        # res[-1].'.'
        source_lang = 'en'
        target_lang = 'hi'
        full_report = understand_report(res, source_lang, target_lang)

        returnValue = res + "," + "Important Keywords: "+","+keys+"," + full_report

        # print(all_possible_meaing)
        return returnValue
    return None

if __name__ == '__main__':
    app.run()
