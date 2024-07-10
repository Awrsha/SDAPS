from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'aPzY3lK4u5Q7w9x8W2e0nR3sV5mU7vN2'

# Load models
first_model = load_model('static/Models/FirstModel.h5')
second_model = load_model('static/Models/SecondModel.h5')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def get_recommendations(prediction):
    benign_recommendations = [
        "استفاده از کرم\u200cهای مرطوب\u200cکننده مانند کرم اوسرین یا کرم نیتروژنا.",
        "استفاده از کرم\u200cهای ضد التهاب مانند هیدروکورتیزون یا بتامتازون.",
        "استفاده از قرص\u200cهای آنتی\u200cهیستامین مانند لوراتادین یا سیتریزین برای کاهش خارش.",
        "استفاده از کرم\u200cهای ضد آفتاب با SPF بالا مانند کرم ضد آفتاب الارو یا کرم ضد آفتاب آردن.",
        "خودداری از تماس مستقیم با مواد تحریک\u200cکننده و آلرژن\u200cها."
    ]
    
    malignant_recommendations = [
        "مراجعه فوری به متخصص پوست و سرطان\u200cشناس برای بررسی دقیق و تعیین برنامه درمانی مناسب.",
        "استفاده از داروهای ضد سرطان مانند فلوروراسیل (5-FU) یا ایمیکیمود (Aldara).",
        "استفاده از کرم\u200cهای ضد التهاب مانند کلوبتازول یا مومتازون برای مدیریت علائم پوستی.",
        "استفاده از کرم\u200cهای مرطوب\u200cکننده مانند کرم اوسرین یا کرم نیتروژنا برای تسکین خشکی و تحریک پوست.",
        "استفاده از پمادهای ضد عفونی کننده و آنتی بیوتیک مانند موپیروسین برای جلوگیری از عفونت.",
        "مراجعه به روان\u200cشناس یا مشاور برای مدیریت استرس و اضطراب."
    ]
    
    if prediction < 0.5:  # Benign
        return benign_recommendations
    else:  # Malignant
        return malignant_recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Preprocess the image
            preprocessed_image = preprocess_image(file_path)
            
            # Make predictions
            pred1 = first_model.predict(preprocessed_image)[0][0]
            pred2 = second_model.predict(preprocessed_image)[0][0]
            
            # Calculate average prediction
            avg_pred = (pred1 + pred2) / 2
            
            # Get recommendations
            recommendations = get_recommendations(avg_pred)
            
            # Render the result template
            return render_template('result.html', 
                                   filename=filename,
                                   pred1=pred1*100, 
                                   pred2=pred2*100, 
                                   avg_pred=avg_pred*100,
                                   recommendations=recommendations)
        else:
            flash('Allowed file types are png, jpg, jpeg, gif')
            return redirect(request.url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
