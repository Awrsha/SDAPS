from flask import Flask, jsonify, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import requests
import base64

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = tf.keras.models.load_model("static/models/model_version_1.keras")

GROQ_API_KEY = "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"
HUGGINGFACE_TOKEN = "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"

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

def get_medication_recommendations(skin_type, symptoms, severity, age, gender, medical_history, affected_area, duration):
    common_medications = {
        "خشک": ["هیدروکورتیزون ۱٪", "سماکلوبتازول", "کرم مرطوب کننده CeraVe", "اوسرین", "کرم ویتامین E موضعی"],
        "چرب": ["کلیندامایسین موضعی", "آداپالن ژل", "بنزوئیل پروکساید ۲.۵٪", "لوسیون لاروش پوزای افلوم", "کرم ضدآفتاب فاقد چربی SVR"],
        "معمولی": ["کرم ترتینوئین ۰.۰۲۵٪", "هیدروکینون ۴٪", "کرم مرطوب کننده نیوآ", "آزلائیک اسید ۲۰٪", "کرم ضدآفتاب سینره"],
        "مختلط": ["ژل شستشوی سبامد", "لوسیون ضدجوش اکنئوتین", "کرم مرطوب کننده نئوتروژینا", "کلیندامایسین و ترتینوئین ترکیبی", "تونر پاک کننده پریم"],
        "حساس": ["کرم کالامین", "پماد زینک اکساید", "کرم آلوئه ورا خالص", "پانتنول موضعی", "سرم آب‌رسان اوردینری"],
    }
    
    age_based_recommendations = {
        "child": ["پماد کالاندولا", "کرم هیدروکورتیزون ۰.۵٪", "لوسیون ضد خارش QV", "کرم اوسرین بی‌بی"],
        "teen": ["صابون گوگرد", "ژل بنزوئیل پروکساید ۵٪", "لوسیون سالیسیلیک اسید ۲٪", "پن کلیندامایسین"],
        "adult": ["کرم ترتینوئین ۰.۰۵٪", "کرم هیدروکینون ۴٪", "سرم ویتامین C", "آزلائیک اسید ۲۰٪"],
        "elderly": ["کرم اوره ۱۰٪", "پماد وازلین", "کرم مرطوب کننده قوی آرتودرم", "کرم ویتامین D موضعی"]
    }
    
    condition_based_recommendations = {
        "خارش": ["قرص سیتریزین", "کرم کالامین", "لوسیون کالامین", "قرص هیدروکسیزین", "پماد دیفن‌هیدرامین"],
        "قرمزی": ["کرم هیدروکورتیزون", "کرم پیمکرولیموس", "ژل آلوئه ورا", "کرم سینک اکساید", "قرص فکسوفنادین"],
        "تورم": ["قرص ایبوپروفن", "پماد دگزامتازون", "کمپرس سرد", "قرص سلکوکسیب", "ژل دیکلوفناک"],
        "درد": ["قرص استامینوفن", "ژل لیدوکائین موضعی ۲٪", "پماد کاپسایسین", "قرص ناپروکسن", "پچ پیروکسیکام"],
        "خشکی": ["کرم اوره ۲۰٪", "پماد وازلین", "روغن بادام موضعی", "کرم لاکتیک اسید", "کرم گلیسیرین"],
        "پوسته‌ریزی": ["شامپو کتوکونازول", "کرم سالیسیلیک اسید", "لوسیون اوره", "روغن درخت چای", "کرم ضد قارچ"],
        "تاول": ["پماد موپیروسین", "محلول پرمنگنات پتاسیم", "پماد باسیتراسین", "کرم سولفادیازین نقره", "کرم فوسیدیک اسید"],
        "سوزش": ["ژل آلوئه ورا", "اسپری پانتنول", "کرم کالامین", "لوسیون کالامین", "کمپرس خنک"],
        "تغییر رنگ": ["کرم هیدروکینون", "کرم ترتینوئین", "سرم ویتامین C", "کرم آزلائیک اسید", "کرم آربوتین"]
    }
    
    medical_history_considerations = {
        "دیابت": ["از استروئیدهای قوی اجتناب شود", "استفاده از محصولات بدون الکل", "کرم‌های ملایم و بدون عطر"],
        "فشار خون بالا": ["اجتناب از کورتیکواستروئیدهای سیستمیک", "مراقبت از تداخلات دارویی"],
        "آلرژی": ["محصولات هیپوآلرژنیک", "تست پچ قبل از استفاده از داروهای جدید", "اجتناب از عطرها"],
        "بیماری خود ایمنی": ["مراقبت ویژه برای جلوگیری از تشدید علائم", "مشورت با متخصص روماتولوژی قبل از درمان‌های جدید"],
        "سابقه سرطان پوست": ["معاینات منظم پوست", "استفاده از ضد آفتاب قوی", "اجتناب از ترکیبات فوتوسنسیتیو"]
    }
    
    area_specific_recommendations = {
        "صورت": ["از محصولات غیر کومدوژنیک استفاده شود", "کرم‌های ملایم‌تر با غلظت کمتر", "ضد آفتاب روزانه"],
        "اندام": ["کرم‌های مرطوب کننده قوی‌تر", "درمان‌های اکلوزیو شبانه"],
        "تنه": ["فرمولاسیون‌های سبک‌تر برای سطوح وسیع", "لوسیون به جای کرم برای پوشش بهتر"],
        "سر": ["شامپوهای درمانی مخصوص", "محلول‌های موضعی به جای کرم"]
    }
    
    recommendations = []
    
    if skin_type in common_medications:
        recommendations.extend(common_medications[skin_type][:2])
    
    age_category = "adult"
    if age < 12:
        age_category = "child"
    elif age < 20:
        age_category = "teen"
    elif age > 65:
        age_category = "elderly"
    
    recommendations.extend(age_based_recommendations[age_category][:2])
    
    for symptom in symptoms:
        if symptom in condition_based_recommendations and symptom != "هیچ کدام":
            recommendations.extend(condition_based_recommendations[symptom][:1])
    
    for condition in medical_history:
        if condition in medical_history_considerations and condition != "هیچ کدام":
            recommendations.append(medical_history_considerations[condition][0])
    
    for area in [affected_area]:
        if area in area_specific_recommendations:
            recommendations.append(area_specific_recommendations[area][0])
    
    if severity > 5:
        recommendations.append("مراجعه به متخصص پوست در اسرع وقت")
    
    duration_advice = ""
    if duration == "چند روز":
        duration_advice = "در صورت عدم بهبود پس از یک هفته، به پزشک مراجعه کنید"
    elif duration == "چند هفته":
        duration_advice = "مراجعه به متخصص پوست توصیه می‌شود"
    elif duration == "چند ماه" or duration == "چند سال":
        duration_advice = "نیاز به درمان تخصصی و بررسی دقیق دارد"
    
    if duration_advice:
        recommendations.append(duration_advice)
    
    return recommendations

def get_ai_recommendation(user_data, prediction, image_path=None):
    try:
        if prediction > 0.6:
            condition = "بدخیم"
            malignancy_level = "بالا"
        elif prediction > 0.3:
            condition = "مشکوک"
            malignancy_level = "متوسط"
        else:
            condition = "خوش خیم"
            malignancy_level = "پایین"

        age_group = "سالمند" if user_data['age'] > 65 else "بزرگسال" if user_data['age'] > 18 else "نوجوان" if user_data['age'] > 12 else "کودک"
        
        symptoms_text = ", ".join([s for s in user_data['symptoms'] if s != "هیچ کدام"]) if user_data['symptoms'] else "بدون علامت خاص"
        
        medical_history_text = ", ".join([m for m in user_data['medical_history'] if m != "هیچ کدام"]) if user_data['medical_history'] else "بدون سابقه پزشکی خاص"
        
        system_prompt = """
        شما یک سیستم تشخیص پوستی هستید که به ارائه توصیه های دقیق و تخصصی می‌پردازد. لطفا یک تشخیص احتمالی و توصیه‌های درمانی مناسب ارائه دهید.
        توصیه‌های شما باید شامل موارد زیر باشد:
        1. تشخیص احتمالی بر اساس اطلاعات ارائه شده
        2. توصیه‌های درمانی شامل داروهای موضعی، خوراکی و مراقبت‌های عمومی
        3. اقدامات پیشگیرانه
        4. توصیه به مراجعه به پزشک در صورت نیاز

        پاسخ خود را به صورت کامل و تخصصی به زبان فارسی ارائه دهید.
        همیشه در پایان تاکید کنید که تشخیص نهایی باید توسط پزشک متخصص انجام شود.
        """

        user_message = f"""
        اطلاعات بیمار:
        - سن: {user_data['age']} ({age_group})
        - جنسیت: {user_data['gender']}
        - نوع پوست: {user_data['skin_type']}
        - قومیت: {user_data['ethnicity']}
        - علائم: {symptoms_text}
        - شدت علائم: {user_data['symptom_severity']}/10
        - ناحیه درگیر: {user_data['affected_area']}
        - سابقه پزشکی: {medical_history_text}
        - مدت زمان علائم: {user_data['symptom_duration']}
        - اطلاعات تکمیلی: {user_data['additional_info']}
        
        نتیجه تشخیص هوش مصنوعی:
        - وضعیت: {condition}
        - احتمال بدخیمی: {prediction * 100:.1f}%
        - سطح خطر: {malignancy_level}
        
        لطفاً بر اساس این اطلاعات، تشخیص احتمالی و توصیه‌های درمانی دقیق و تخصصی ارائه دهید.
        """

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "model": "llama3-70b-8192",
            "temperature": 0.2,
            "max_tokens": 1024
        }
        
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                               headers=headers, 
                               json=payload)
        
        if response.status_code == 200:
            ai_response = response.json()["choices"][0]["message"]["content"]
            return ai_response
        else:
            app.logger.error(f"Error from Groq API: {response.text}")
            # Fallback to local recommendation if AI service fails
            recommendations = get_medication_recommendations(
                user_data['skin_type'], 
                user_data['symptoms'], 
                user_data['symptom_severity'],
                user_data['age'],
                user_data['gender'],
                user_data['medical_history'],
                user_data['affected_area'],
                user_data['symptom_duration']
            )
            
            if prediction > 0.6:
                condition = "بدخیم"
                general_advice = "لطفاً هر چه سریعتر به پزشک متخصص مراجعه کنید."
            elif prediction > 0.3:
                condition = "مشکوک"
                general_advice = "توصیه می‌شود برای بررسی دقیق‌تر به متخصص پوست مراجعه کنید."
            else:
                condition = "خوش خیم"
                general_advice = "به نظر می رسد مشکل جدی نیست، اما برای اطمینان با پزشک مشورت کنید."
            
            personal_advice = "توصیه‌های درمانی بر اساس اطلاعات شما:\n- " + "\n- ".join(recommendations)
            personal_advice += "\n\nاین توصیه‌ها جایگزین مراجعه به پزشک نمی‌شود و صرفاً جنبه اطلاع‌رسانی دارد."
            
            return f"وضعیت: {condition}\n\n{general_advice}\n\n{personal_advice}"
    except Exception as e:
        app.logger.error(f"Error getting AI recommendation: {e}")
        # Ultimate fallback to a simple recommendation
        if prediction > 0.6:
            return "احتمال بدخیمی بالاست. لطفاً هر چه سریعتر به پزشک متخصص پوست مراجعه کنید."
        else:
            return "احتمال بدخیمی پایین است. اما برای اطمینان، با پزشک متخصص پوست مشورت کنید."

def analyze_image_with_huggingface(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}"
        }
        
        payload = {
            "inputs": {
                "image": encoded_image
            }
        }
        
        api_url = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            # Extract the top 3 classifications
            top_predictions = []
            for pred in result[:3]:
                label = pred.get("label", "").replace("_", " ").title()
                confidence = pred.get("score", 0) * 100
                top_predictions.append(f"{label} ({confidence:.1f}%)")
            
            return ", ".join(top_predictions)
        else:
            app.logger.error(f"Error from HuggingFace API: {response.text}")
            return None
    except Exception as e:
        app.logger.error(f"Error analyzing image with HuggingFace: {e}")
        return None

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

        huggingface_analysis = analyze_image_with_huggingface(file_path)
        
        formatted_prediction = f"{prediction * 100:.0f}%"
        
        if prediction > 0.6:
            condition = "بدخیم"
        elif prediction > 0.4:
            condition = "مشکوک"
        else:
            condition = "خوش خیم"
            
        ai_recommendation = get_ai_recommendation(user_data, prediction, file_path)
        general_advice, personal_advice = "", ""
        
        if ai_recommendation:
            parts = ai_recommendation.split("\n\n")
            if len(parts) >= 3:
                general_advice = parts[1]
                personal_advice = "\n".join(parts[2:])
            else:
                general_advice = "لطفاً برای بررسی دقیق‌تر به پزشک متخصص مراجعه کنید."
                personal_advice = ai_recommendation

        html = render_template('result.html', 
                               condition=condition, 
                               prediction=formatted_prediction,
                               general_advice=general_advice, 
                               personal_advice=personal_advice,
                               image_filename=filename,
                               huggingface_analysis=huggingface_analysis)
        return jsonify({'html': html})
    else:
        app.logger.error('File type not allowed')
        return jsonify({'error': 'نوع فایل مجاز نیست.'})

if __name__ == '__main__':
    app.run(debug=True)