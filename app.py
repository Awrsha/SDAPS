from flask import Flask, jsonify, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load your pre-trained model
model = tf.keras.models.load_model("static/models/model_version_1.keras")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def clear_upload_folder():
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
    for f in files:
        os.remove(f)

def predict_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            prediction = model.predict(img_array)
            return prediction[0][0]
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return None

def get_recommendation(prediction, user_data):
    if prediction > 0.6:
        condition = "بدخیم"
        general_advice = "لطفاً هر چه سریعتر به پزشک متخصص مراجعه کنید."
    else:
        condition = "خوش خیم"
        general_advice = "به نظر می رسد مشکل جدی نیست، اما برای اطمینان با پزشک مشورت کنید."

    personal_advice = "با توجه به اطلاعات شما: "
    if user_data['age'] > 50:
        personal_advice += "با توجه به سن شما، مراقبت های پوستی ویژه توصیه می شود. "
    if user_data['skin_type'] == 'خشک':
        personal_advice += "از کرم های مرطوب کننده قوی استفاده کنید. "
    if 'دیابت' in user_data['medical_history']:
        personal_advice += "به دلیل سابقه دیابت، مراقبت های پوستی خاص نیاز است. "
    if user_data['symptom_severity'] > 7:
        personal_advice += "با توجه به شدت علائم، مراجعه فوری به پزشک توصیه می‌شود. "
    if 'سر' in user_data['affected_area']:
        personal_advice += "برای ضایعات پوستی در ناحیه سر، مراقبت‌های ویژه‌ای لازم است. "
    
    return condition, general_advice, personal_advice

@app.route('/')
def index():
    clear_upload_folder()
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    app.logger.info('Received request for diagnosis')
    app.logger.info(f'Files in request: {request.files}')
    app.logger.info(f'Form data: {request.form}')

    if 'file' not in request.files:
        app.logger.error('No file part in the request')
        return jsonify({'error': 'لطفاً یک تصویر انتخاب کنید.'})
     
    file = request.files['file']
    
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'فایلی انتخاب نشده است.'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        user_data = {
            'age': int(request.form.get('age', 0)),
            'skin_type': request.form.get('skin_type', ''),
            'gender': request.form.get('gender', ''),
            'ethnicity': request.form.get('ethnicity', ''),
            'symptoms': request.form.getlist('symptoms'),
            'symptom_severity': int(request.form.get('symptom_severity', 1)),
            'affected_area': request.form.get('affected_area', ''),
            'medical_history': request.form.getlist('medical_history'),
            'symptom_duration': request.form.get('symptom_duration', ''),
            'additional_info': request.form.get('additional_info', '')
        }
        
        app.logger.info(f'Processed user data: {user_data}')

        prediction = predict_image(file_path)
        if prediction is None:
            return jsonify({'error': 'پردازش تصویر با مشکل مواجه شد.'})

        formatted_prediction = f"{prediction * 100:.0f}%"

        condition, general_advice, personal_advice = get_recommendation(prediction, user_data)

        app.logger.info(f"Condition: {condition}")
        app.logger.info(f"General Advice: {general_advice}")
        app.logger.info(f"Personal Advice: {personal_advice}")

        html = render_template('result.html', 
                               condition=condition, 
                               prediction=formatted_prediction,
                               general_advice=general_advice, 
                               personal_advice=personal_advice,
                               image_filename=filename) 
        return jsonify({'html': html})
    else:
        app.logger.error('File type not allowed')
        return jsonify({'error': 'نوع فایل مجاز نیست.'})

if __name__ == '__main__':
    app.run(debug=True)