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
import random
from datetime import datetime

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = tf.keras.models.load_model("D:/ML , NN/skin-4/static/models/best_model.keras")

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
            # First input: Extract 71 features
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            # Flatten and take first 71 features
            flattened = img_array.reshape(1, -1)
            features_71 = flattened[:, :71]
            
            # Second input: 128x128 image
            img_128 = img.resize((128, 128))
            img_array_128 = np.array(img_128)
            img_array_128 = np.expand_dims(img_array_128, axis=0)
            img_array_128 = img_array_128 / 255.0
            
            # Make prediction with named inputs in dictionary format
            prediction = model.predict({
                'features': features_71,
                'images': img_array_128
            })
            return prediction[0][0]
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return None

def analyze_with_huggingface(image_path):
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
            return response.json()
        else:
            app.logger.error(f"Error from HuggingFace API: {response.text}")
            return None
    except Exception as e:
        app.logger.error(f"Error analyzing image with HuggingFace: {e}")
        return None

def get_dermatology_diseases():
    common_skin_diseases = {
        "اگزما": {
            "description": "یک بیماری التهابی پوستی مزمن که با قرمزی، خارش و پوسته‌پوسته شدن پوست مشخص می‌شود.",
            "symptoms": ["خارش", "قرمزی", "پوسته‌ریزی", "خشکی پوست"],
            "type": "التهابی",
            "risk_level": "کم",
            "common_locations": ["صورت", "آرنج‌ها", "زانوها", "مچ‌ها"]
        },
        "پسوریازیس": {
            "description": "یک بیماری خود ایمنی که باعث رشد سریع سلول‌های پوستی و ایجاد پلاک‌های قرمز با پوسته‌های نقره‌ای می‌شود.",
            "symptoms": ["پوسته‌ریزی", "قرمزی", "خارش", "پلاک‌های ضخیم"],
            "type": "التهابی",
            "risk_level": "متوسط",
            "common_locations": ["آرنج‌ها", "زانوها", "پوست سر", "کمر پایین"]
        },
        "آکنه": {
            "description": "یک بیماری التهابی فولیکول‌های مو و غدد چربی که با جوش‌ها، کومدون‌ها و التهاب مشخص می‌شود.",
            "symptoms": ["جوش", "کومدون", "التهاب", "درد"],
            "type": "التهابی",
            "risk_level": "کم",
            "common_locations": ["صورت", "پشت", "سینه"]
        },
        "رزاسه": {
            "description": "یک بیماری پوستی مزمن که با قرمزی صورت، رگ‌های برجسته و گاهی جوش‌های شبیه آکنه مشخص می‌شود.",
            "symptoms": ["قرمزی", "التهاب", "رگ‌های برجسته", "جوش"],
            "type": "التهابی",
            "risk_level": "کم",
            "common_locations": ["گونه‌ها", "بینی", "پیشانی", "چانه"]
        },
        "درماتیت سبورئیک": {
            "description": "یک بیماری التهابی مزمن که با پوسته‌های چرب و قرمزی در مناطق غنی از غدد چربی مشخص می‌شود.",
            "symptoms": ["پوسته‌ریزی چرب", "قرمزی", "خارش"],
            "type": "التهابی",
            "risk_level": "کم",
            "common_locations": ["پوست سر", "صورت", "گوش‌ها", "سینه"]
        },
        "لیکن پلان": {
            "description": "یک بیماری التهابی که با ضایعات بنفش‌رنگ براق و خطوط سفید روی آنها مشخص می‌شود.",
            "symptoms": ["خارش", "ضایعات بنفش", "خطوط سفید"],
            "type": "التهابی",
            "risk_level": "متوسط",
            "common_locations": ["مچ‌ها", "پاها", "دهان", "ناخن‌ها"]
        },
        "کهیر": {
            "description": "یک واکنش آلرژیک که با برآمدگی‌های قرمز و خارش‌دار روی پوست مشخص می‌شود.",
            "symptoms": ["خارش", "برآمدگی‌های قرمز", "تورم"],
            "type": "آلرژیک",
            "risk_level": "کم",
            "common_locations": ["تمام بدن"]
        },
        "عفونت قارچی": {
            "description": "عفونت‌های ناشی از قارچ‌ها که با ضایعات قرمز، پوسته‌پوسته و خارش‌دار مشخص می‌شوند.",
            "symptoms": ["خارش", "قرمزی", "پوسته‌ریزی", "سوزش"],
            "type": "عفونی",
            "risk_level": "کم",
            "common_locations": ["پاها", "کشاله ران", "ناخن‌ها"]
        },
        "زرد زخم": {
            "description": "یک عفونت باکتریایی سطحی پوست که با تاول‌ها و پوسته‌های زرد عسلی مشخص می‌شود.",
            "symptoms": ["تاول", "پوسته‌های زرد", "قرمزی"],
            "type": "عفونی",
            "risk_level": "متوسط",
            "common_locations": ["صورت", "دست‌ها", "پاها"]
        },
        "کارسینوم سلول بازال": {
            "description": "شایع‌ترین نوع سرطان پوست که با برآمدگی‌های براق یا زخم‌های بدون بهبود مشخص می‌شود.",
            "symptoms": ["برآمدگی براق", "زخم بدون بهبود", "رگ‌های سطحی"],
            "type": "بدخیم",
            "risk_level": "بالا",
            "common_locations": ["صورت", "گوش‌ها", "گردن", "مناطق در معرض آفتاب"]
        },
        "کارسینوم سلول سنگفرشی": {
            "description": "نوعی سرطان پوست که از سلول‌های سنگفرشی منشأ می‌گیرد و با زخم‌ها یا برآمدگی‌های قرمز و فلسی مشخص می‌شود.",
            "symptoms": ["برآمدگی قرمز", "زخم", "پوسته‌ریزی"],
            "type": "بدخیم",
            "risk_level": "بالا",
            "common_locations": ["لب‌ها", "گوش‌ها", "صورت", "دست‌ها"]
        },
        "ملانوما": {
            "description": "خطرناک‌ترین نوع سرطان پوست که از سلول‌های ملانوسیت منشأ می‌گیرد.",
            "symptoms": ["خال نامتقارن", "حاشیه نامنظم", "تغییر رنگ", "قطر بزرگ"],
            "type": "بدخیم",
            "risk_level": "بالا",
            "common_locations": ["تمام بدن، به‌ویژه مناطق در معرض آفتاب"]
        },
        "کراتوز آکتینیک": {
            "description": "ضایعات پیش‌سرطانی ناشی از آسیب نور خورشید که می‌توانند به سرطان پوست تبدیل شوند.",
            "symptoms": ["پوسته‌های زبر", "قرمزی", "خارش گاهی"],
            "type": "پیش‌سرطانی",
            "risk_level": "متوسط تا بالا",
            "common_locations": ["صورت", "لب‌ها", "گوش‌ها", "پوست سر در افراد کم‌مو"]
        }
    }
    return common_skin_diseases

def get_diagnostic_features(disease_name):
    features = {
        "اگزما": ["اریتم (قرمزی)", "پوسته‌ریزی", "خراش", "خشکی پوست", "ضایعات پراکنده", "مرز نامشخص"],
        "پسوریازیس": ["پلاک‌های برجسته", "پوسته‌های نقره‌ای", "خطوط جداکننده واضح", "ضخیم شدگی پوست", "اریتم زمینه‌ای"],
        "آکنه": ["کومدون‌های باز و بسته", "پاپول‌ها و پوستول‌ها", "ندول‌ها", "اسکار احتمالی", "چربی پوست"],
        "رزاسه": ["اریتم مرکزی صورت", "تلانژکتازی", "پاپول‌ها و پوستول‌ها", "فلاشینگ", "ادم"],
        "درماتیت سبورئیک": ["پوسته‌های چرب", "اریتم خفیف", "ضایعات در نواحی چرب پوست", "خارش متغیر"],
        "لیکن پلان": ["پاپول‌های بنفش براق", "خطوط ویکهام", "ضایعات چندوجهی", "توزیع دوطرفه"],
        "کهیر": ["ویل‌های برجسته", "اریتم اطراف", "تغییر مکان ضایعات", "محو شدن در فشار"],
        "عفونت قارچی": ["حاشیه مشخص و فعال", "پوسته‌ریزی", "وزیکول‌های کوچک", "ضایعه حلقوی"],
        "زرد زخم": ["پوسته‌های زرد عسلی", "اریتم زمینه‌ای", "تاول‌های سطحی", "ترشح چرکی"],
        "کارسینوم سلول بازال": ["پاپول براق", "تلانژکتازی سطحی", "حاشیه مرواریدی", "اولسراسیون مرکزی", "رشد آهسته"],
        "کارسینوم سلول سنگفرشی": ["پلاک هایپرکراتوتیک", "اولسراسیون", "پایه عریض", "رشد سریع", "خونریزی آسان"],
        "ملانوما": ["عدم تقارن", "حاشیه نامنظم", "رنگ ناهمگن", "قطر بیش از 6mm", "تغییر در زمان", "عروق نامنظم"],
        "کراتوز آکتینیک": ["ماکول یا پاپول زبر", "پوسته‌های چسبنده", "رنگ قرمز-قهوه‌ای", "سطح خشن"]
    }
    
    default_features = ["اریتم (قرمزی)", "پوسته‌ریزی", "ضایعات پراکنده", "مرز نامشخص", "التهاب موضعی"]
    return features.get(disease_name, default_features)

def get_treatments(disease_name, patient_data):
    base_treatments = {
        "اگزما": [
            {"type": "موضعی", "treatment": "کورتیکواستروئید با قدرت متوسط (تریامسینولون 0.1%)", "priority": "بالا"},
            {"type": "موضعی", "treatment": "مهارکننده کلسی‌نورین (تاکرولیموس 0.03% یا 0.1%)", "priority": "متوسط"},
            {"type": "موضعی", "treatment": "مرطوب کننده‌های غیر کومدوژنیک", "priority": "بالا"},
            {"type": "سیستمیک", "treatment": "آنتی‌هیستامین‌های خوراکی (سرترالین)", "priority": "متوسط"},
            {"type": "سایر", "treatment": "اجتناب از محرک‌های شناخته شده", "priority": "بالا"}
        ],
        "پسوریازیس": [
            {"type": "موضعی", "treatment": "کورتیکواستروئید قوی (کلوبتازول 0.05%)", "priority": "بالا"},
            {"type": "موضعی", "treatment": "آنالوگ‌های ویتامین D (کلسیپوتریول)", "priority": "بالا"},
            {"type": "موضعی", "treatment": "رتینوئیدها (تازاروتن)", "priority": "متوسط"},
            {"type": "سیستمیک", "treatment": "متوترکسات", "priority": "متوسط"},
            {"type": "بیولوژیک", "treatment": "مهارکننده‌های TNF-α (اتانرسپت، اینفلیکسیماب)", "priority": "پایین"},
            {"type": "سایر", "treatment": "فتوتراپی UVB باریک‌باند", "priority": "متوسط"}
        ],
        "آکنه": [
            {"type": "موضعی", "treatment": "رتینوئیدها (ترتینوئین، آداپالن)", "priority": "بالا"},
            {"type": "موضعی", "treatment": "بنزوئیل پراکساید", "priority": "بالا"},
            {"type": "موضعی", "treatment": "آنتی‌بیوتیک‌های موضعی (کلیندامایسین، اریترومایسین)", "priority": "متوسط"},
            {"type": "سیستمیک", "treatment": "آنتی‌بیوتیک‌های خوراکی (داکسی‌سایکلین، مینوسایکلین)", "priority": "متوسط"},
            {"type": "سیستمیک", "treatment": "ایزوترتینوئین خوراکی", "priority": "بالا (در موارد شدید)"},
            {"type": "هورمونی", "treatment": "قرص‌های ضدبارداری (در خانم‌ها)", "priority": "متوسط"}
        ],
        "رزاسه": [
            {"type": "موضعی", "treatment": "مترونیدازول 0.75% یا 1%", "priority": "بالا"},
            {"type": "موضعی", "treatment": "آزلائیک اسید 15% یا 20%", "priority": "بالا"},
            {"type": "موضعی", "treatment": "ایوِرمکتین 1%", "priority": "متوسط"},
            {"type": "سیستمیک", "treatment": "داکسی‌سایکلین (40mg) یا مینوسایکلین", "priority": "متوسط"},
            {"type": "سایر", "treatment": "اجتناب از محرک‌های شناخته شده (الکل، غذاهای تند، نور خورشید)", "priority": "بالا"},
            {"type": "سایر", "treatment": "درمان با لیزر برای تلانژکتازی", "priority": "پایین"}
        ],
        "درماتیت سبورئیک": [
            {"type": "موضعی", "treatment": "شامپو و کرم ضد قارچ (کتوکونازول 2%)", "priority": "بالا"},
            {"type": "موضعی", "treatment": "کورتیکواستروئید با قدرت کم تا متوسط", "priority": "متوسط"},
            {"type": "موضعی", "treatment": "مهارکننده‌های کلسی‌نورین (پیمکرولیموس)", "priority": "متوسط"},
            {"type": "سیستمیک", "treatment": "در موارد مقاوم: ایتراکونازول", "priority": "پایین"},
            {"type": "سایر", "treatment": "کنترل استرس", "priority": "متوسط"}
        ],
        "کارسینوم سلول بازال": [
            {"type": "جراحی", "treatment": "اکسیزیون کامل با مارژین مناسب", "priority": "بالا"},
            {"type": "جراحی", "treatment": "جراحی میکروگرافیک موس", "priority": "بالا"},
            {"type": "موضعی", "treatment": "ایمیکیمود 5% (در موارد سطحی)", "priority": "متوسط"},
            {"type": "سایر", "treatment": "رادیوتراپی (در موارد خاص یا بیماران مسن)", "priority": "متوسط"},
            {"type": "سایر", "treatment": "کرایوتراپی (در موارد سطحی کوچک)", "priority": "متوسط"},
            {"type": "سایر", "treatment": "فتودینامیک تراپی", "priority": "متوسط"}
        ],
        "کارسینوم سلول سنگفرشی": [
            {"type": "جراحی", "treatment": "اکسیزیون وسیع با مارژین مناسب", "priority": "بالا"},
            {"type": "جراحی", "treatment": "بیوپسی غدد لنفاوی سنتینل در موارد خاص", "priority": "متوسط تا بالا"},
            {"type": "سایر", "treatment": "رادیوتراپی (کمکی یا در موارد غیرقابل جراحی)", "priority": "متوسط"},
            {"type": "سیستمیک", "treatment": "شیمی‌درمانی (در موارد متاستاتیک)", "priority": "متوسط"},
            {"type": "سایر", "treatment": "پیگیری منظم پس از درمان", "priority": "بالا"}
        ],
        "ملانوما": [
            {"type": "جراحی", "treatment": "اکسیزیون وسیع با مارژین استاندارد براساس ضخامت تومور", "priority": "بالا"},
            {"type": "جراحی", "treatment": "بیوپسی غدد لنفاوی سنتینل", "priority": "بالا"},
            {"type": "سیستمیک", "treatment": "ایمونوتراپی (پمبرولیزومب، نیوولومب)", "priority": "بالا (در مراحل پیشرفته)"},
            {"type": "سیستمیک", "treatment": "درمان هدفمند (در موارد جهش BRAF)", "priority": "بالا (در مراحل پیشرفته)"},
            {"type": "سایر", "treatment": "پیگیری دقیق و منظم", "priority": "بالا"}
        ],
        "کراتوز آکتینیک": [
            {"type": "موضعی", "treatment": "5-فلورواوراسیل 5%", "priority": "بالا"},
            {"type": "موضعی", "treatment": "ایمیکیمود 5%", "priority": "بالا"},
            {"type": "سایر", "treatment": "کرایوتراپی", "priority": "بالا"},
            {"type": "سایر", "treatment": "تراپی فوتودینامیک", "priority": "متوسط"},
            {"type": "موضعی", "treatment": "دیکلوفناک سدیم 3% در ژل هیالورونیک اسید", "priority": "متوسط"},
            {"type": "سایر", "treatment": "محافظت از آفتاب و پیگیری منظم", "priority": "بالا"}
        ]
    }
    
    default_treatments = [
        {"type": "موضعی", "treatment": "کورتیکواستروئید با قدرت مناسب", "priority": "متوسط"},
        {"type": "موضعی", "treatment": "مرطوب کننده‌های مناسب", "priority": "بالا"},
        {"type": "سیستمیک", "treatment": "آنتی‌هیستامین‌های خوراکی", "priority": "متوسط"},
        {"type": "سایر", "treatment": "اصلاح سبک زندگی و رژیم غذایی", "priority": "متوسط"},
        {"type": "سایر", "treatment": "پیگیری منظم با متخصص پوست", "priority": "بالا"}
    ]
    
    treatments = base_treatments.get(disease_name, default_treatments)
    
    # تنظیم درمان‌ها بر اساس شرایط بیمار
    if 'دیابت' in patient_data.get('medical_history', []):
        treatments = [t for t in treatments if not 
                     (t['treatment'].startswith('کورتیکواستروئید قوی') and t['priority'] == 'بالا')]
        treatments.append({
            "type": "توجه ویژه", 
            "treatment": "استفاده محتاطانه از کورتیکواستروئیدها با توجه به دیابت", 
            "priority": "بالا"
        })
    
    if patient_data.get('age', 0) > 65:
        treatments = [t for t in treatments if not
                     (t['treatment'].startswith('متوترکسات') and t['priority'] == 'بالا')]
        
        if any(t['type'] == 'سیستمیک' for t in treatments):
            treatments.append({
                "type": "توجه ویژه",
                "treatment": "تنظیم دوز داروهای سیستمیک با توجه به سن بیمار",
                "priority": "بالا"
            })
    
    return treatments

def get_prevention_methods(diseases):
    base_methods = {
        "اگزما": [
            "استفاده از مرطوب کننده‌های بدون عطر به صورت منظم",
            "اجتناب از محرک‌های شناخته شده (صابون‌های قوی، مواد شوینده)",
            "استفاده از لباس‌های نخی و اجتناب از پارچه‌های مصنوعی",
            "حفظ رطوبت محیط، به خصوص در فصول سرد",
            "مدیریت استرس که می‌تواند باعث تشدید علائم شود"
        ],
        "پسوریازیس": [
            "حفظ وزن سالم و مدیریت استرس",
            "مرطوب نگه داشتن پوست به طور منظم",
            "اجتناب از آسیب‌های پوستی (پدیده کوبنر)",
            "محدود کردن مصرف الکل و دخانیات",
            "مراقبت از پوست در فصول سرد و خشک"
        ],
        "آکنه": [
            "شستشوی منظم پوست با شوینده‌های ملایم",
            "استفاده از محصولات غیر کومدوژنیک",
            "پرهیز از دستکاری ضایعات",
            "مدیریت استرس",
            "رژیم غذایی متعادل و محدود کردن مصرف لبنیات و قندهای ساده در موارد تشدیدکننده"
        ],
        "رزاسه": [
            "اجتناب از محرک‌های شناخته شده (نور خورشید، غذاهای تند، الکل)",
            "استفاده روزانه از ضد آفتاب مناسب برای پوست حساس",
            "استفاده از محصولات آرایشی و بهداشتی غیر کومدوژنیک",
            "کنترل دمای محیط و اجتناب از گرمای شدید",
            "شستشوی ملایم صورت با شوینده‌های ملایم"
        ],
        "کارسینوم سلول بازال": [
            "استفاده روزانه از ضد آفتاب با SPF حداقل 30",
            "پوشیدن لباس‌های محافظ و کلاه در برابر آفتاب",
            "اجتناب از قرار گرفتن در معرض نور خورشید در ساعات اوج (10 صبح تا 4 بعدازظهر)",
            "معاینه منظم پوست توسط خود فرد و پزشک",
            "شناسایی و درمان به موقع ضایعات مشکوک"
        ],
        "کارسینوم سلول سنگفرشی": [
            "استفاده روزانه از ضد آفتاب با SPF حداقل 30",
            "پوشیدن لباس‌های محافظ و کلاه در برابر آفتاب",
            "اجتناب از قرار گرفتن در معرض نور خورشید در ساعات اوج (10 صبح تا 4 بعدازظهر)",
            "معاینه منظم پوست به دنبال ضایعات مشکوک",
            "درمان زودهنگام کراتوز آکتینیک"
        ],
        "ملانوما": [
            "استفاده روزانه از ضد آفتاب با SPF حداقل 30",
            "پوشیدن لباس‌های محافظ و کلاه در برابر آفتاب",
            "اجتناب از قرار گرفتن در معرض نور خورشید در ساعات اوج (10 صبح تا 4 بعدازظهر)",
            "بررسی منظم خال‌ها با قانون ABCDE",
            "معاینه سالیانه پوست توسط متخصص"
        ],
        "کراتوز آکتینیک": [
            "استفاده روزانه از ضد آفتاب با SPF حداقل 30",
            "پوشیدن لباس‌های محافظ و کلاه در برابر آفتاب",
            "اجتناب از قرار گرفتن در معرض نور خورشید در ساعات اوج (10 صبح تا 4 بعدازظهر)",
            "معاینه منظم پوست توسط متخصص پوست",
            "درمان زودهنگام ضایعات مشکوک"
        ]
    }
    
    common_methods = [
        "رعایت بهداشت پوست و شستشوی منظم با شوینده‌های ملایم",
        "استفاده از مرطوب کننده‌های مناسب برای نوع پوست",
        "محافظت در برابر نور خورشید با استفاده از ضد آفتاب",
        "رژیم غذایی متعادل و مصرف کافی آب",
        "اجتناب از استرس و حفظ سبک زندگی سالم"
    ]
    
    top_disease = diseases[0]['name'] if diseases else None
    
    if top_disease and top_disease in base_methods:
        methods = base_methods[top_disease]
    else:
        methods = common_methods
    
    return methods

def get_suggested_tests(diseases, patient_data):
    base_tests = {
        "کارسینوم سلول بازال": [
            "بیوپسی پوست (پانچ، شیو یا اکسیژنال)",
            "بررسی پاتولوژیک ضایعه",
            "درماتوسکوپی دیجیتال",
            "در صورت نیاز: اسکن CT یا PET برای ارزیابی متاستاز",
            "آزمایش ژنتیک مولکولی در موارد خاص"
        ],
        "کارسینوم سلول سنگفرشی": [
            "بیوپسی پوست",
            "بررسی پاتولوژیک ضایعه",
            "ارزیابی غدد لنفاوی ناحیه‌ای",
            "در موارد پیشرفته: CT اسکن یا PET-CT",
            "پیگیری منظم پس از درمان"
        ],
        "ملانوما": [
            "بیوپسی اکسیژنال کامل",
            "بررسی پاتولوژیک با تعیین ضخامت برسلو",
            "بیوپسی غدد لنفاوی سنتینل در موارد ضخامت بیش از 1mm",
            "تصویربرداری (CT اسکن، MRI، PET-CT) در موارد پیشرفته",
            "آزمایش‌های مولکولی برای بررسی جهش BRAF"
        ],
        "پسوریازیس": [
            "بیوپسی پوست در موارد تشخیصی مبهم",
            "آزمایش خون برای بررسی مارکرهای التهابی (ESR, CRP)",
            "در موارد مشکوک به آرتریت پسوریاتیک: رادیوگرافی مفاصل",
            "بررسی پروفایل چربی و قند خون (ریسک فاکتورهای همراه)",
            "ارزیابی شاخص PASI برای تعیین شدت بیماری"
        ],
        "اگزما": [
            "تست پچ برای شناسایی آلرژن‌های احتمالی",
            "آزمایش IgE سرم در موارد مشکوک به آتوپی",
            "بیوپسی پوست در موارد تشخیصی مبهم"
        ],
        "عفونت قارچی": [
            "آزمایش مستقیم KOH",
            "کشت قارچ",
            "بیوپسی در موارد مقاوم یا غیرمعمول",
            "تست‌های سرولوژیک در موارد سیستمیک مشکوک"
        ]
    }
    
    default_tests = [
        "معاینه بالینی دقیق توسط متخصص پوست",
        "درماتوسکوپی برای بررسی دقیق‌تر ضایعه",
        "بیوپسی پوست در صورت نیاز به تشخیص قطعی",
        "آزمایشات خونی پایه (CBC, ESR)"
    ]
    
    top_disease = diseases[0]['name'] if diseases else None
    
    if top_disease and top_disease in base_tests:
        tests = base_tests[top_disease]
    else:
        tests = default_tests
    
    # تنظیم آزمایشات پیشنهادی بر اساس شرایط بیمار
    if 'دیابت' in patient_data.get('medical_history', []):
        tests.append('بررسی کنترل قند خون و HbA1c')
    
    if 'سابقه سرطان پوست' in patient_data.get('medical_history', []):
        tests.append('بررسی کامل پوست برای شناسایی ضایعات جدید')
        tests.append('درماتوسکوپی دیجیتال و مقایسه با تصاویر قبلی')
    
    return tests

def generate_diagnostic_findings(diseases, patient_data):
    top_disease = diseases[0]['name'] if diseases else None
    
    if top_disease == "اگزما":
        findings = f"تصویر نشان‌دهنده ضایعات اریتماتو با پوسته‌ریزی و حدود نامشخص است. با توجه به محل ضایعه در {patient_data.get('affected_area', '')} و علائم بالینی شامل {', '.join(patient_data.get('symptoms', []))}, تشخیص اگزما محتمل‌ترین گزینه است. این تشخیص با سابقه {patient_data.get('symptom_duration', '')} مطابقت دارد."
    elif top_disease == "پسوریازیس":
        findings = f"ضایعات پلاک‌مانند با پوسته‌های نقره‌ای و حدود مشخص در {patient_data.get('affected_area', '')} مشاهده می‌شود. این یافته‌ها همراه با علائم {', '.join(patient_data.get('symptoms', []))} به شدت مطرح‌کننده پسوریازیس است. مدت {patient_data.get('symptom_duration', '')} علائم با ماهیت مزمن این بیماری همخوانی دارد."
    elif top_disease == "رزاسه":
        findings = f"اریتم منتشر در ناحیه مرکزی صورت همراه با تلانژکتازی و پاپول‌های التهابی مشهود است. با توجه به محل درگیری و علائم {', '.join(patient_data.get('symptoms', []))}, تشخیص رزاسه در اولویت قرار دارد. این بیماری معمولاً در بزرگسالان بروز می‌کند که با سن بیمار ({patient_data.get('age', '')}) مطابقت دارد."
    elif "کارسینوم" in top_disease or top_disease == "ملانوما":
        findings = f"ضایعه‌ای با حدود نامنظم و رنگ ناهمگن در {patient_data.get('affected_area', '')} دیده می‌شود. ویژگی‌های تصویری شامل عدم تقارن، تغییرات سطحی و عروق غیرطبیعی است که مطرح‌کننده {top_disease} می‌باشد. با توجه به سن بیمار ({patient_data.get('age', '')}) و سابقه {patient_data.get('symptom_duration', '')} علائم، بررسی دقیق‌تر پاتولوژیک توصیه می‌شود."
    else:
        findings = f"تصویر نشان‌دهنده ضایعات پوستی در ناحیه {patient_data.get('affected_area', '')} است که با علائم {', '.join(patient_data.get('symptoms', []))} همراه است. یافته‌های بالینی و تصویری مطرح‌کننده {top_disease} به عنوان اولین تشخیص افتراقی است. با توجه به مدت زمان {patient_data.get('symptom_duration', '')} علائم و سابقه پزشکی بیمار، بررسی‌های تکمیلی توصیه می‌گردد."
    
    return findings

def generate_treatment_approach(diseases, patient_data):
    top_disease = diseases[0] if diseases else None
    
    if not top_disease:
        return "با توجه به اطلاعات محدود، رویکرد درمانی دقیق نیازمند معاینه بالینی است."
    
    disease_name = top_disease['name']
    disease_type = top_disease.get('type', '')
    
    if disease_type == 'بدخیم':
        approach = f"با توجه به احتمال بالای بیماری {disease_name} که یک ضایعه {disease_type} است، رویکرد درمانی باید شامل ارجاع فوری به متخصص پوست و انجام بیوپسی باشد. پس از تأیید تشخیص، ممکن است جراحی اکسیزیون با مارژین مناسب، رادیوتراپی یا درمان‌های مودالیته دیگر براساس مرحله و نوع دقیق ضایعه نیاز باشد."
    elif disease_type == 'پیش‌سرطانی':
        approach = f"ضایعه مطرح شده {disease_name} که از نوع {disease_type} است، نیازمند توجه و درمان دقیق می‌باشد. بررسی بیشتر توسط متخصص پوست و احتمالاً بیوپسی برای تأیید تشخیص توصیه می‌شود. درمان‌های موضعی مانند کرایوتراپی، کورتاژ، 5-فلورواوراسیل یا ایمیکیمود بسته به شدت و گستردگی ضایعات قابل تجویز است."
    elif disease_type == 'التهابی':
        approach = f"برای درمان {disease_name} که یک بیماری {disease_type} است، معمولاً ترکیبی از کنترل عوامل محرک، مراقبت‌های پوستی و درمان‌های موضعی کاربرد دارد. با توجه به شدت علائم ({patient_data.get('symptom_severity', '')}/10) و مدت زمان آن ({patient_data.get('symptom_duration', '')}), کنترل التهاب و بهبود مانع پوستی از اهداف اصلی درمان است."
    elif disease_type == 'عفونی':
        approach = f"برای درمان {disease_name} که یک عامل {disease_type} است، استفاده از داروهای ضد قارچی یا ضد میکروبی مناسب توصیه می‌شود. با توجه به گستردگی ضایعه و مدت زمان {patient_data.get('symptom_duration', '')}, ممکن است درمان‌های موضعی یا سیستمیک نیاز باشد. کنترل فاکتورهای مستعدکننده عفونت نیز اهمیت زیادی دارد."
    elif disease_type == 'خودایمنی':
        approach = f"{disease_name} به عنوان یک بیماری {disease_type}، نیازمند رویکرد چندجانبه شامل کنترل سیستم ایمنی، کاهش التهاب و مراقبت‌های پوستی است. با توجه به سابقه پزشکی بیمار و شدت علائم ({patient_data.get('symptom_severity', '')}/10)، ارزیابی کامل سیستمیک و درمان‌های اختصاصی توسط متخصصین پوست و روماتولوژی توصیه می‌شود."
    else:
        approach = f"برای {disease_name} رویکرد درمانی شامل کنترل علائم، بهبود کیفیت زندگی و جلوگیری از پیشرفت بیماری است. با توجه به شدت علائم ({patient_data.get('symptom_severity', '')}/10) و مدت زمان آن ({patient_data.get('symptom_duration', '')}), ترکیبی از درمان‌های موضعی، سیستمیک و تغییرات سبک زندگی توصیه می‌شود."
    
    # اضافه کردن ملاحظات خاص بر اساس سابقه پزشکی
    if 'دیابت' in patient_data.get('medical_history', []):
        approach += ' با توجه به سابقه دیابت، ملاحظات ویژه‌ای در انتخاب درمان‌ها و پایش بهبود زخم‌ها باید مدنظر قرار گیرد.'
    
    if 'بیماری خود ایمنی' in patient_data.get('medical_history', []):
        approach += ' سابقه بیماری خودایمنی نیازمند هماهنگی نزدیک بین متخصصان پوست و روماتولوژی در مدیریت درمان است.'
    
    return approach

def calculate_follow_up_priority(diseases, patient_data):
    top_disease = diseases[0] if diseases else None
    
    if not top_disease:
        return "متوسط", 0.5
    
    disease_type = top_disease.get('type', '')
    probability = top_disease.get('probability', 0)
    
    if disease_type == 'بدخیم' or probability > 0.6:
        priority_level = "فوری"
        priority_value = 0.9
    elif disease_type == 'پیش‌سرطانی' or probability > 0.4:
        priority_level = "زودهنگام"
        priority_value = 0.7
    else:
        priority_level = "معمولی"
        priority_value = 0.4
    
    # تنظیم بر اساس سن و شدت علائم
    age = int(patient_data.get('age', 0))
    if age > 70:
        priority_value = min(priority_value + 0.1, 0.95)
    
    symptom_severity = int(patient_data.get('symptom_severity', 5))
    if symptom_severity >= 8:
        priority_value = min(priority_value + 0.1, 0.95)
    
    return priority_level, priority_value

def generate_follow_up_recommendation(priority_level):
    if priority_level == "فوری":
        return "توصیه می‌شود بیمار در اسرع وقت (ترجیحاً طی 24 تا 48 ساعت آینده) به متخصص پوست مراجعه نماید."
    elif priority_level == "زودهنگام":
        return "توصیه می‌شود بیمار طی هفته آینده به متخصص پوست مراجعه نماید."
    else:
        return "پیگیری معمول توصیه می‌شود. بیمار می‌تواند طی دو هفته آینده به متخصص پوست مراجعه نماید."

def generate_diagnosis_results(patient_data, prediction, image_filename, huggingface_results=None):
    # تنظیم احتمال‌های اولیه
    base_probability = prediction
    
    # دریافت لیست بیماری‌های پوستی
    skin_diseases = get_dermatology_diseases()
    
    # انتخاب تصادفی چند بیماری برای تشخیص افتراقی
    potential_diagnoses = []
    
    # تعیین بیماری‌های احتمالی بر اساس علائم و ناحیه درگیر
    location = patient_data.get('affected_area', '')
    symptoms = patient_data.get('symptoms', [])
    age = int(patient_data.get('age', 30))
    
    # بررسی بیماری‌های مرتبط با محل و علائم
    for disease_name, disease_info in skin_diseases.items():
        score = 0.0
        
        # امتیاز بر اساس محل درگیری
        if location and location in disease_info['common_locations']:
            score += 0.2
        
        # امتیاز بر اساس علائم
        common_symptoms = set(symptoms) & set(disease_info['symptoms'])
        if common_symptoms:
            score += len(common_symptoms) * 0.15
        
        # امتیاز بر اساس سن (برای برخی بیماری‌ها)
        if disease_name == "آکنه" and 15 <= age <= 30:
            score += 0.1
        elif disease_name == "رزاسه" and age > 30:
            score += 0.1
        elif disease_name in ["کارسینوم سلول بازال", "کارسینوم سلول سنگفرشی", "کراتوز آکتینیک"] and age > 50:
            score += 0.2
        
        # سابقه پزشکی
        medical_history = patient_data.get('medical_history', [])
        if "سابقه سرطان پوست" in medical_history and disease_info['type'] in ["بدخیم", "پیش‌سرطانی"]:
            score += 0.3
        
        # اضافه کردن به لیست با احتمال محاسبه شده
        if score > 0:
            # تنظیم وزن مدل با نتیجه پیش‌بینی اصلی
            if disease_info['type'] == "بدخیم" and base_probability > 0.5:
                score += 0.4
            elif disease_info['type'] != "بدخیم" and base_probability < 0.3:
                score += 0.3
            
            potential_diagnoses.append({
                'name': disease_name,
                'score': score,
                'type': disease_info['type'],
                'risk_level': disease_info['risk_level'],
                'description': disease_info['description'],
                'symptoms': disease_info['symptoms'],
            })
    
    # اضافه کردن حداقل یک بیماری بدخیم برای مقایسه
    has_malignant = any(d['type'] == "بدخیم" for d in potential_diagnoses)
    if not has_malignant:
        for disease_name, disease_info in skin_diseases.items():
            if disease_info['type'] == "بدخیم":
                potential_diagnoses.append({
                    'name': disease_name,
                    'score': max(0.1, base_probability - 0.2),
                    'type': disease_info['type'],
                    'risk_level': disease_info['risk_level'],
                    'description': disease_info['description'],
                    'symptoms': disease_info['symptoms'],
                })
                break
    
    # مرتب‌سازی و محدود کردن تعداد تشخیص‌ها
    potential_diagnoses = sorted(potential_diagnoses, key=lambda x: x['score'], reverse=True)
    top_diagnoses = potential_diagnoses[:min(5, len(potential_diagnoses))]
    
    # نرمال‌سازی احتمالات به گونه‌ای که مجموع آنها 1 شود
    total_score = sum(d['score'] for d in top_diagnoses)
    for diagnosis in top_diagnoses:
        diagnosis['probability'] = diagnosis['score'] / total_score if total_score > 0 else 0
    
    # ویژگی‌های تصویری بیماری اصلی
    image_features = get_diagnostic_features(top_diagnoses[0]['name']) if top_diagnoses else []
    
    # گزینه‌های درمانی
    treatment_options = get_treatments(top_diagnoses[0]['name'], patient_data) if top_diagnoses else []
    
    # تشخیص یافته‌ها
    diagnostic_findings = generate_diagnostic_findings(top_diagnoses, patient_data)
    
    # رویکرد درمانی
    treatment_approach = generate_treatment_approach(top_diagnoses, patient_data)
    
    # توصیه‌های پیشگیری
    prevention_methods = get_prevention_methods(top_diagnoses)
    
    # آزمایشات پیشنهادی
    suggested_tests = get_suggested_tests(top_diagnoses, patient_data)
    
    # اولویت پیگیری
    priority_level, priority_value = calculate_follow_up_priority(top_diagnoses, patient_data)
    follow_up_recommendation = generate_follow_up_recommendation(priority_level)
    
    # اگر نتایج HuggingFace موجود باشد، از آن استفاده کنیم
    ai_detected_classes = []
    if huggingface_results:
        for result in huggingface_results[:3]:
            label = result.get("label", "").replace("_", " ").title()
            confidence = result.get("score", 0) * 100
            ai_detected_classes.append(f"{label} ({confidence:.1f}%)")
    
    diagnosis_results = {
        'diagnoses': top_diagnoses,
        'diagnostic_type': top_diagnoses[0]['type'] if top_diagnoses else None,
        'image_features': image_features,
        'treatment_options': treatment_options,
        'diagnostic_findings': diagnostic_findings,
        'treatment_approach': treatment_approach,
        'prevention_methods': prevention_methods,
        'suggested_tests': suggested_tests,
        'follow_up_priority': {
            'level': priority_level,
            'value': priority_value,
            'recommendation': follow_up_recommendation
        },
        'image_filename': image_filename,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'ai_detected_classes': ai_detected_classes
    }
    
    return diagnosis_results

@app.route('/')
def index():
    clear_upload_folder()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'success': True, 'filename': filename})
    
    return jsonify({'success': False, 'message': 'File type not allowed'})

@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        app.logger.info('Received request for diagnosis')
        app.logger.info(f'Form data: {request.form}')

        if 'file' not in request.files and 'filename' not in request.form:
            app.logger.error('No file part in the request')
            return jsonify({'error': 'لطفاً یک تصویر انتخاب کنید.'})
        
        # اگر فایل در درخواست باشد آن را ذخیره کنیم
        if 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                app.logger.error('No selected file')
                return jsonify({'error': 'فایلی انتخاب نشده است.'})
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
            else:
                return jsonify({'error': 'نوع فایل مجاز نیست.'})
        else:
            # استفاده از نام فایل موجود
            filename = request.form['filename']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            if not os.path.exists(file_path):
                return jsonify({'error': 'فایل مورد نظر یافت نشد.'})
        
        # استخراج اطلاعات بیمار از فرم
        patient_data = {
            'age': int(request.form.get('age', 0)),
            'gender': request.form.get('gender', ''),
            'skin_type': request.form.get('skin_type', ''),
            'ethnicity': request.form.get('ethnicity', ''),
            'symptoms': request.form.getlist('symptoms[]') if request.form.getlist('symptoms[]') else request.form.get('symptoms', '').split(','),
            'symptom_severity': int(request.form.get('symptom_severity', 5)),
            'affected_area': request.form.get('affected_area', ''),
            'medical_history': request.form.getlist('medical_history[]') if request.form.getlist('medical_history[]') else request.form.get('medical_history', '').split(','),
            'symptom_duration': request.form.get('symptom_duration', ''),
            'additional_info': request.form.get('additional_info', ''),
            'patient_history': request.form.get('patient_history', ''),
            'current_medications': request.form.get('current_medications', ''),
            'drug_allergies': request.form.get('drug_allergies', ''),
            'previous_treatments': request.form.get('previous_treatments', '')
        }
        
        app.logger.info(f'Processed patient data: {patient_data}')

        # پیش‌بینی تصویر
        prediction = predict_image(file_path)
        if prediction is None:
            return jsonify({'error': 'پردازش تصویر با مشکل مواجه شد.'})
        
        app.logger.info(f'Prediction result: {prediction}')
        
        # تحلیل اضافی با HuggingFace (اختیاری)
        huggingface_results = analyze_with_huggingface(file_path)
        
        # تولید نتایج تشخیص
        diagnosis_results = generate_diagnosis_results(patient_data, prediction, filename, huggingface_results)
        
        app.logger.info(f'Generated diagnosis results: {diagnosis_results}')
        
        # رندر کردن HTML صفحه نتایج
        html = render_template(
            'result.html',
            patient_data=patient_data,
            diagnosis_results=diagnosis_results,
            image_url=f'/static/uploads/{filename}'
        )
        
        return jsonify({
            'success': True,
            'html': html,
            'prediction': float(prediction),
            'diagnosis': diagnosis_results
        })
    
    except Exception as e:
        app.logger.error(f'Error in diagnosis: {str(e)}')
        return jsonify({'error': f'خطای سیستمی: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)