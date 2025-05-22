<h1 align="center">
  <br>
  SDAPS: Smart Dermatological Analysis & Prescription System
</h1>

<p align="center">
  <em>Revolutionizing dermatological diagnostics with AI-powered image analysis and personalized treatment recommendations.</em>
</p>

<p align="center">
  <!-- Replace with your actual badges -->
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/tensorflow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/LLM-LLAMA3_Local-green.svg" alt="LLM">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
  <img src="https://img.shields.io/github/stars/your-username/sdaps-repo?style=social" alt="GitHub Stars">
</p>

---

## 🌟 Executive Abstract

The Smart Dermatological Analysis & Prescription System (SDAPS) represents a groundbreaking integration of computer vision and natural language processing technologies to revolutionize dermatological diagnostics and treatment. This web-based platform employs a multi-modal approach combining advanced deep learning models with clinical data analysis to provide accurate skin lesion classification and personalized treatment recommendations.

SDAPS addresses critical healthcare challenges including dermatologist shortages, limited access to specialized care, and delayed diagnosis of potentially serious skin conditions. By leveraging attention-based convolutional neural networks (ResNet-50 with Gate Attention) for image analysis and local large language models (LLMs) for medical text processing, SDAPS achieves **Best AUC: 0.92570** across a diverse range of skin conditions.

---

## ✨ Key Features

*   🧠 **Dual-Modal Analysis Pipeline**: Integrates visual processing of dermatological images (CNNs) with textual analysis of patient data (LLMs) for holistic diagnosis.
*   🔬 **Advanced Neural Architecture**: Utilizes ResNet-50 with Gate Attention mechanisms to focus on critical visual features in skin lesions.
*   💊 **Personalized Medicine Approach**: Incorporates patient-specific factors (age, gender, medical history, symptoms) via LLM for tailored treatment plans.
*   🇮🇷 **Localized Pharmaceutical Database**: Recommends practical treatments using locally available and approved medications in Iran (with global applicability).
*   🗺️ **Interactive Body Mapping**: Enhances spatial precision for documenting and analyzing lesion locations.
*   💡 **Diagnostic Aid**: Designed to support dermatologists, not replace them, improving efficiency and reach.

---

## ⚙️ How It Works

SDAPS employs an innovative dual-pipeline architecture to process both visual and textual information:

1.  **Image Upload & Analysis**:
    *   User uploads a skin lesion image.
    *   The image is processed by the **ResNet-50 with Gate Attention** model.
    *   Key visual features are extracted, focusing on diagnostically relevant areas.
2.  **Patient Data Input**:
    *   User inputs patient demographics, medical history, reported symptoms, and lesion location (optionally via interactive body map).
3.  **LLM Processing**:
    *   The **Local LLM (LLAMA3)** processes the textual patient data.
    *   It contextualizes symptoms and history.
4.  **Multi-Modal Fusion & Diagnosis**:
    *   Outputs from the image analysis and text processing pipelines are combined.
    *   The system generates diagnostic suggestions with confidence scores.
5.  **Personalized Prescription**:
    *   Based on the diagnosis and patient data, the LLM queries its localized pharmaceutical database.
    *   It suggests a personalized treatment plan, considering locally available medications and potential drug interactions.
