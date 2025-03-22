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
from llama_index.core.tools import FunctionTool

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

# Define functions to be used as tools
def get_basic_skin_recommendations(skin_type: str, age: int) -> str:
    """
    Get basic skin care recommendations based on skin type and age.
    
    Args:
        skin_type: Type of skin (خشک, چرب, معمولی, مختلط, حساس)
        age: Age of the patient
        
    Returns:
        String containing basic skin care recommendations
    """
    recommendations = {
        "خشک": {
            "child": "استفاده از کرم‌های مرطوب کننده مخصوص کودکان بدون عطر و مواد حساسیت‌زا",
            "teen": "استفاده از کرم‌های مرطوب کننده حاوی سرامید و هیالورونیک اسید",
            "adult": "استفاده از کرم‌های غنی حاوی روغن‌های طبیعی، سرامید و اسکوالان",
            "elderly": "استفاده از کرم‌های بسیار غنی حاوی اوره، گلیسیرین و روغن‌های طبیعی"
        },
        "چرب": {
            "child": "استفاده از ژل‌های شستشوی ملایم مخصوص کودکان",
            "teen": "استفاده از محصولات حاوی سالیسیلیک اسید و بنزوئیل پراکساید",
            "adult": "استفاده از فوم‌های پاک‌کننده حاوی نیاسینامید و لوسیون‌های غیرچرب",
            "elderly": "استفاده از محصولات متعادل‌کننده چربی با فرمولاسیون ملایم‌تر"
        },
        "معمولی": {
            "child": "استفاده از محصولات ملایم مناسب برای کودکان",
            "teen": "استفاده از ژل‌های شستشو و مرطوب‌کننده‌های سبک",
            "adult": "استفاده از سرم‌های آنتی‌اکسیدان و مرطوب‌کننده‌های متعادل",
            "elderly": "استفاده از کرم‌های ضد پیری و مرطوب‌کننده‌های غنی‌تر"
        },
        "مختلط": {
            "child": "استفاده از محصولات ملایم و متعادل‌کننده مخصوص کودکان",
            "teen": "استفاده از ژل‌های متعادل‌کننده T-zone و مرطوب‌کننده‌های سبک",
            "adult": "استفاده از محصولات متعادل‌کننده چربی و مرطوب‌کننده‌های نواحی خشک",
            "elderly": "استفاده از محصولات چندمنظوره و مرطوب‌کننده‌های مناسب هر ناحیه"
        },
        "حساس": {
            "child": "استفاده از محصولات بدون عطر و مواد حساسیت‌زا مخصوص پوست حساس کودکان",
            "teen": "استفاده از محصولات هیپوآلرژنیک و آرام‌بخش پوست",
            "adult": "استفاده از محصولات حاوی سنتلا آسیاتیکا، آلانتوئین و پانتنول",
            "elderly": "استفاده از محصولات بسیار ملایم، فاقد عطر و حاوی مواد ترمیم‌کننده"
        }
    }
    
    age_category = "adult"
    if age < 12:
        age_category = "child"
    elif age < 20:
        age_category = "teen"
    elif age > 65:
        age_category = "elderly"
    
    if skin_type in recommendations and age_category in recommendations[skin_type]:
        return recommendations[skin_type][age_category]
    else:
        return "استفاده از محصولات مناسب با نوع پوست و مشورت با متخصص پوست"

def get_symptom_treatments(symptoms: list) -> str:
    """
    Get treatment recommendations based on symptoms.
    
    Args:
        symptoms: List of symptoms (خارش, قرمزی, تورم, درد, etc.)
        
    Returns:
        String containing treatment recommendations for the symptoms
    """
    treatments = {
        "خارش": "استفاده از کرم‌های حاوی هیدروکورتیزون 1% یا کرم کالامین، مصرف آنتی‌هیستامین‌ها مانند سیتریزین یا لوراتادین",
        "قرمزی": "استفاده از کرم‌های حاوی آزلائیک اسید، نیاسینامید یا عصاره سنتلا آسیاتیکا. اجتناب از محرک‌ها و استفاده از آب سرد",
        "تورم": "استفاده از کمپرس سرد، مصرف داروهای ضدالتهاب غیراستروئیدی مانند ایبوپروفن (با مشورت پزشک) و استفاده از ژل‌های حاوی آرنیکا",
        "درد": "استفاده از کمپرس سرد یا گرم (بسته به نوع ضایعه)، مصرف مسکن‌های خوراکی با مشورت پزشک، و استفاده از ژل‌های موضعی حاوی لیدوکائین",
        "خشکی": "استفاده از کرم‌های مرطوب‌کننده غنی حاوی سرامید، هیالورونیک اسید، اوره یا گلیسیرین. نوشیدن آب کافی و استفاده از دستگاه بخور",
        "پوسته‌ریزی": "استفاده از کرم‌های حاوی اوره، لاکتیک اسید یا سالیسیلیک اسید. لایه‌برداری ملایم و استفاده از شامپوهای ضد شوره در صورت درگیری سر",
        "تاول": "اجتناب از ترکاندن تاول‌ها، استفاده از پانسمان‌های هیدروکلوئید، پماد آنتی‌بیوتیک موضعی در صورت عفونت و محافظت از ناحیه",
        "سوزش": "استفاده از ژل آلوئه ورا، کمپرس سرد، اسپری‌های حاوی منتول یا کالامین و اجتناب از مواد محرک",
        "تغییر رنگ": "استفاده از کرم‌های حاوی هیدروکینون (با مشورت پزشک)، آزلائیک اسید، ویتامین C یا اسید کوجیک. محافظت از نور آفتاب"
    }
    
    result = []
    for symptom in symptoms:
        if symptom in treatments and symptom != "هیچ کدام":
            result.append(f"{symptom}: {treatments[symptom]}")
    
    if result:
        return "\n".join(result)
    else:
        return "توصیه‌های خاصی برای علائم شما وجود ندارد. مشورت با متخصص پوست توصیه می‌شود."

def assess_malignancy_risk(prediction_value: float, age: int, medical_history: list) -> str:
    """
    Assess the risk of malignancy based on prediction value, age and medical history.
    
    Args:
        prediction_value: AI prediction value (0-1)
        age: Age of the patient
        medical_history: List of medical history conditions
        
    Returns:
        String containing risk assessment and recommendations
    """
    risk_level = "پایین"
    recommendations = []
    
    if prediction_value > 0.7:
        risk_level = "بسیار بالا"
        recommendations.append("مراجعه فوری به متخصص پوست (حداکثر طی 24-48 ساعت آینده)")
        recommendations.append("انجام بیوپسی برای تشخیص قطعی ضایعه")
    elif prediction_value > 0.5:
        risk_level = "بالا"
        recommendations.append("مراجعه به متخصص پوست در اسرع وقت (طی هفته آینده)")
        recommendations.append("بررسی ضایعه با درماتوسکوپی")
    elif prediction_value > 0.3:
        risk_level = "متوسط"
        recommendations.append("مراجعه به متخصص پوست در یک ماه آینده")
        recommendations.append("پیگیری و مقایسه تغییرات ضایعه")
    else:
        risk_level = "پایین"
        recommendations.append("پیگیری ضایعه از نظر تغییرات هر 3-6 ماه")
        recommendations.append("عکسبرداری منظم برای مقایسه")
    
    # Age considerations
    if age > 65:
        recommendations.append("با توجه به سن بالا، بررسی دقیق‌تر توصیه می‌شود")
    elif age < 18 and prediction_value > 0.3:
        recommendations.append("با توجه به سن پایین و احتمال بدخیمی، بررسی تخصصی ضروری است")
    
    # Medical history considerations
    if "سابقه سرطان پوست" in medical_history:
        recommendations.append("با توجه به سابقه سرطان پوست، نیاز به بررسی تخصصی و دقیق‌تر است")
        if prediction_value > 0.2:  # Lower threshold for those with history
            risk_level = "بالا" if risk_level != "بسیار بالا" else risk_level
    
    if "بیماری خود ایمنی" in medical_history and prediction_value > 0.3:
        recommendations.append("با توجه به بیماری خود ایمنی، مشورت با متخصص ایمونولوژی نیز توصیه می‌شود")
    
    return f"سطح خطر: {risk_level}\n\nتوصیه‌ها:\n- " + "\n- ".join(recommendations)

def get_skincare_routine(skin_type: str, age: int, condition: str) -> str:
    """
    Get a daily skincare routine based on skin type, age and condition.
    
    Args:
        skin_type: Type of skin
        age: Age of the patient
        condition: Current skin condition (خوش خیم, مشکوک, بدخیم)
        
    Returns:
        String containing daily skincare routine
    """
    cleanser = {
        "خشک": "شوینده کرمی ملایم (مانند سرم ست آبرسان)",
        "چرب": "ژل یا فوم شستشو حاوی سالیسیلیک اسید (مانند ست اویل فری)",
        "معمولی": "شوینده متعادل غیر صابونی (مانند سرم ست)",
        "مختلط": "ژل شستشوی ملایم برای پوست مختلط (مانند سرم ست ترکیبی)",
        "حساس": "شوینده بدون عطر و ملایم (مانند سرم ست پوست حساس)"
    }
    
    moisturizer = {
        "خشک": "کرم مرطوب‌کننده غنی حاوی سرامید (مانند سرم ست هیدرا)",
        "چرب": "ژل مرطوب‌کننده سبک فاقد چربی (مانند مرطوب‌کننده ژلی نیوتروژینا)",
        "معمولی": "کرم یا لوسیون مرطوب‌کننده متعادل (مانند سرم ست آبرسان)",
        "مختلط": "ژل-کرم مرطوب‌کننده (مانند هیدرودرم)",
        "حساس": "مرطوب‌کننده بدون عطر مخصوص پوست حساس (مانند اوسرین یا کیکومید)"
    }
    
    sunscreen = {
        "خشک": "کرم ضدآفتاب مرطوب‌کننده SPF50 (مانند سینره یا فیوژن واتر)",
        "چرب": "ضدآفتاب فاقد چربی یا ژلی SPF50 (مانند سان سنس یا SVR)",
        "معمولی": "ضدآفتاب با فرمولاسیون متعادل SPF50 (مانند لاروش پوزای)",
        "مختلط": "ضدآفتاب سبک و غیرچرب SPF50 (مانند هیدرودرم)",
        "حساس": "ضدآفتاب مخصوص پوست حساس SPF50+ (مانند ایزی سان یا آوبن)"
    }
    
    special_care = ""
    if condition == "مشکوک" or condition == "بدخیم":
        special_care = """
توجه ویژه:
- از دستکاری ضایعات پوستی اجتناب کنید
- از قرار گرفتن در معرض آفتاب خودداری کنید
- ضدآفتاب را هر 2 ساعت تجدید کنید
- از محصولات حاوی الکل و مواد محرک استفاده نکنید
- از خراشیدن یا فشار دادن ضایعات خودداری کنید"""
    
    s_type = skin_type if skin_type in cleanser else "معمولی"
    
    routine = f"""برنامه روزانه مراقبت از پوست:

صبح:
1. شستشو: {cleanser[s_type]}
2. مرطوب‌کننده: {moisturizer[s_type]}
3. ضدآفتاب: {sunscreen[s_type]}

شب:
1. شستشو: {cleanser[s_type]}
2. مرطوب‌کننده: {moisturizer[s_type]}

{special_care}

توجه: در صورت استفاده از داروهای تجویزی، آنها را طبق دستور پزشک مصرف کنید و این برنامه را با توصیه‌های پزشک خود تطبیق دهید.
"""
    return routine

def recommend_common_medications(skin_type: str, symptoms: list, severity: int) -> str:
    """
    Recommend common over-the-counter medications based on skin type and symptoms.
    
    Args:
        skin_type: Type of skin
        symptoms: List of symptoms
        severity: Severity of symptoms (1-10)
        
    Returns:
        String containing medication recommendations
    """
    common_medications = {
        "خشک": ["هیدروکورتیزون ۱٪", "سماکلوبتازول", "کرم مرطوب کننده CeraVe", "اوسرین", "کرم ویتامین E موضعی"],
        "چرب": ["کلیندامایسین موضعی", "آداپالن ژل", "بنزوئیل پروکساید ۲.۵٪", "لوسیون لاروش پوزای افلوم", "کرم ضدآفتاب فاقد چربی SVR"],
        "معمولی": ["کرم ترتینوئین ۰.۰۲۵٪", "هیدروکینون ۴٪", "کرم مرطوب کننده نیوآ", "آزلائیک اسید ۲۰٪", "کرم ضدآفتاب سینره"],
        "مختلط": ["ژل شستشوی سبامد", "لوسیون ضدجوش اکنئوتین", "کرم مرطوب کننده نئوتروژینا", "کلیندامایسین و ترتینوئین ترکیبی", "تونر پاک کننده پریم"],
        "حساس": ["کرم کالامین", "پماد زینک اکساید", "کرم آلوئه ورا خالص", "پانتنول موضعی", "سرم آب‌رسان اوردینری"],
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
    
    recommendations = []
    
    # Get skin type specific recommendations
    if skin_type in common_medications:
        recommendations.append(f"داروهای مناسب برای پوست {skin_type}:")
        for med in common_medications[skin_type][:2]:
            recommendations.append(f"- {med}")
    
    # Get symptom specific recommendations
    symptom_meds = []
    for symptom in symptoms:
        if symptom in condition_based_recommendations and symptom != "هیچ کدام":
            for med in condition_based_recommendations[symptom][:1]:
                symptom_meds.append(f"- {med} (برای {symptom})")
    
    if symptom_meds:
        recommendations.append("\nداروهای مناسب برای علائم:")
        recommendations.extend(symptom_meds)
    
    # Severity based recommendations
    if severity > 7:
        recommendations.append("\nبا توجه به شدت بالای علائم:")
        recommendations.append("- مراجعه به پزشک متخصص توصیه می‌شود")
        recommendations.append("- از خوددرمانی پرهیز کنید")
    
    if not recommendations:
        return "داروی خاصی بدون مشورت با پزشک توصیه نمی‌شود. لطفاً با پزشک متخصص مشورت کنید."
    
    return "\n".join(recommendations) + "\n\nتوجه: این توصیه‌ها جنبه عمومی دارند و برای استفاده از هر دارو حتماً با پزشک مشورت کنید."

# Create FunctionTool instances
basic_recommendation_tool = FunctionTool.from_defaults(
    name="get_basic_skin_recommendations",
    description="Get basic skin care recommendations based on skin type and age",
    fn=get_basic_skin_recommendations
)

symptom_treatment_tool = FunctionTool.from_defaults(
    name="get_symptom_treatments",
    description="Get treatment recommendations based on symptoms",
    fn=get_symptom_treatments
)

malignancy_risk_tool = FunctionTool.from_defaults(
    name="assess_malignancy_risk",
    description="Assess the risk of malignancy based on prediction value and patient information",
    fn=assess_malignancy_risk
)

skincare_routine_tool = FunctionTool.from_defaults(
    name="get_skincare_routine",
    description="Get a daily skincare routine based on skin type, age and condition",
    fn=get_skincare_routine
)

medication_tool = FunctionTool.from_defaults(
    name="recommend_common_medications",
    description="Recommend common over-the-counter medications based on skin type and symptoms",
    fn=recommend_common_medications
)

# Create a list of all tools
skin_tools = [
    basic_recommendation_tool,
    symptom_treatment_tool,
    malignancy_risk_tool,
    skincare_routine_tool,
    medication_tool
]

def get_tool_recommendations(user_data, prediction):
    try:
        condition = "بدخیم" if prediction > 0.6 else "مشکوک" if prediction > 0.3 else "خوش خیم"
        
        tool_results = []
        
        # Use the malignancy risk tool
        risk_assessment = malignancy_risk_tool(
            prediction_value=prediction,
            age=user_data['age'],
            medical_history=user_data['medical_history']
        )
        tool_results.append(f"### ارزیابی خطر\n{risk_assessment}")
        
        # Use the basic recommendation tool
        basic_rec = basic_recommendation_tool(
            skin_type=user_data['skin_type'],
            age=user_data['age']
        )
        tool_results.append(f"### توصیه‌های پایه مراقبت پوستی\n{basic_rec}")
        
        # If there are symptoms, use the symptom treatment tool
        if user_data['symptoms'] and not (len(user_data['symptoms']) == 1 and user_data['symptoms'][0] == "هیچ کدام"):
            symptom_rec = symptom_treatment_tool(symptoms=user_data['symptoms'])
            tool_results.append(f"### درمان علائم\n{symptom_rec}")
        
        # Get skincare routine
        routine = skincare_routine_tool(
            skin_type=user_data['skin_type'],
            age=user_data['age'],
            condition=condition
        )
        tool_results.append(f"### برنامه روزانه مراقبت پوستی\n{routine}")
        
        # Get medication recommendations
        medications = medication_tool(
            skin_type=user_data['skin_type'],
            symptoms=user_data['symptoms'],
            severity=user_data['symptom_severity']
        )
        tool_results.append(f"### توصیه‌های دارویی\n{medications}")
        
        # Combine all results
        all_recommendations = "\n\n".join(tool_results)
        
        final_advice = f"""
### وضعیت تشخیص
نتیجه تشخیص هوش مصنوعی نشان می‌دهد که ضایعه پوستی شما احتمالاً {condition} است.
احتمال بدخیمی: {prediction * 100:.1f}%

{all_recommendations}

### توجه مهم
این توصیه‌ها جایگزین مراجعه به پزشک نیست و تنها جنبه آموزشی و اطلاع‌رسانی دارد.
برای تشخیص قطعی و درمان مناسب، حتماً به پزشک متخصص پوست مراجعه کنید.
"""
        
        return final_advice
    except Exception as e:
        app.logger.error(f"Error using tools: {e}")
        return "خطا در پردازش توصیه‌ها. لطفاً به پزشک متخصص مراجعه کنید."

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
        # First try to use the tools for recommendations
        tool_recommendations = get_tool_recommendations(user_data, prediction)
        if tool_recommendations:
            return tool_recommendations
            
        # Fallback to Groq API if tools don't work
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
        elif prediction > 0.3:
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