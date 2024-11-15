import os
import requests
import tempfile
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
import glob
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def load_model():
    model_url = "https://github.com/Awrsha/Intelligent-Skin-Disease-Prediction-System/blob/master/static/models/model_version_1.keras"
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        response = requests.get(model_url)
        temp_file.write(response.content)
        temp_file_path = temp_file.name
    
    model = tf.keras.models.load_model(temp_file_path)
    os.unlink(temp_file_path)
    return model

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
    risk_factors = []
    recommendations = []
    urgency_level = "متوسط"
    
    if prediction > 0.8:
        condition = "بدخیم با ریسک بالا"
        urgency_level = "بسیار فوری"
        recommendations.append("مراجعه فوری به متخصص پوست ضروری است")
    elif prediction > 0.6:
        condition = "بدخیم"
        urgency_level = "فوری"
        recommendations.append("در اسرع وقت به متخصص پوست مراجعه کنید")
    elif prediction > 0.4:
        condition = "مشکوک"
        recommendations.append("برای بررسی دقیق‌تر به متخصص پوست مراجعه کنید")
    else:
        condition = "خوش خیم"
        recommendations.append("به نظر می‌رسد مشکل جدی نیست، اما برای اطمینان معاینه پزشکی توصیه می‌شود")

    age = user_data['age']
    if age > 60:
        risk_factors.append("سن بالا")
        recommendations.append("با توجه به سن بالا، معاینات دوره‌ای پوست هر ۳ ماه یکبار توصیه می‌شود")
    elif age > 40:
        recommendations.append("معاینات دوره‌ای پوست هر ۶ ماه یکبار توصیه می‌شود")

    if user_data['skin_type'] == 'خشک':
        recommendations.extend([
            "استفاده از کرم‌های مرطوب‌کننده حاوی سرامید",
            "پرهیز از شستشوی طولانی مدت با آب داغ",
            "استفاده از پاک‌کننده‌های ملایم بدون صابون"
        ])
    elif user_data['skin_type'] == 'چرب':
        recommendations.extend([
            "استفاده از محصولات غیر کومدوژنیک",
            "پاکسازی منظم پوست با محصولات مناسب",
            "استفاده از مرطوب‌کننده‌های سبک و غیر چرب"
        ])

    medical_history = user_data['medical_history']
    if 'دیابت' in medical_history:
        risk_factors.append("سابقه دیابت")
        recommendations.extend([
            "کنترل منظم قند خون",
            "مراقبت ویژه از زخم‌های پوستی",
            "بررسی روزانه پوست برای یافتن هرگونه تغییر"
        ])
    
    if 'سرطان پوست در خانواده' in medical_history:
        risk_factors.append("سابقه خانوادگی سرطان پوست")
        recommendations.extend([
            "معاینات منظم‌تر پوست",
            "استفاده از ضد آفتاب با SPF حداقل ۵۰",
            "پرهیز از قرار گرفتن طولانی مدت در معرض نور خورشید"
        ])

    symptom_severity = user_data['symptom_severity']
    if symptom_severity > 8:
        urgency_level = "بسیار فوری"
        recommendations.append("با توجه به شدت علائم، مراجعه اورژانسی توصیه می‌شود")
    elif symptom_severity > 6:
        urgency_level = "فوری"
        recommendations.append("مراجعه به پزشک در اولین فرصت ضروری است")

    affected_area = user_data['affected_area']
    high_risk_areas = ['صورت', 'سر', 'گردن']
    if any(area in affected_area for area in high_risk_areas):
        risk_factors.append(f"درگیری ناحیه {affected_area}")
        recommendations.append(f"با توجه به درگیری ناحیه {affected_area}، معاینه تخصصی ضروری است")

    symptom_duration = user_data['symptom_duration']
    if 'بیش از یک سال' in symptom_duration:
        risk_factors.append("طول مدت طولانی علائم")
        recommendations.append("با توجه به مزمن بودن علائم، بررسی دقیق‌تر ضروری است")

    lifestyle_recommendations = [
        "محافظت از پوست در برابر نور خورشید",
        "مصرف کافی آب و مایعات",
        "تغذیه سالم و متعادل",
        "خواب کافی و منظم",
        "کاهش استرس"
    ]

    prevention_tips = [
        "معاینه منظم پوست در منزل",
        "ثبت و پیگیری تغییرات پوستی",
        "استفاده مداوم از ضد آفتاب",
        "پرهیز از قرار گرفتن در معرض اشعه UV"
    ]

    skincare_routine = {
        'صبح': [
            "شستشوی صورت با پاک‌کننده ملایم",
            "استفاده از تونر مناسب نوع پوست",
            "استفاده از سرم ویتامین C",
            "استفاده از مرطوب‌کننده",
            "استفاده از ضد آفتاب"
        ],
        'شب': [
            "پاک کردن آرایش و آلودگی‌ها",
            "شستشوی صورت",
            "استفاده از تونر",
            "استفاده از سرم مناسب",
            "استفاده از مرطوب‌کننده شب"
        ]
    }

    return {
        'condition': condition,
        'urgency_level': urgency_level,
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'lifestyle_recommendations': lifestyle_recommendations,
        'prevention_tips': prevention_tips,
        'skincare_routine': skincare_routine
    }

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

        recommendations = get_recommendation(prediction, user_data)

        html = render_template('result.html',
                             recommendation_data=recommendations,
                             prediction=formatted_prediction,
                             image_filename=filename)
        return jsonify({'html': html})
    else:
        app.logger.error('File type not allowed')
        return jsonify({'error': 'نوع فایل مجاز نیست.'})

if __name__ == '__main__':
    app.run(debug=True)
