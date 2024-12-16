import os
import logging
import pandas as pd
import spacy
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

from sentence_transformers import SentenceTransformer, util
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    roc_curve, 
    roc_auc_score, 
    f1_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
CSV_PATH = 'resume.csv'            # Path to your CSV file
UPLOADED_PDF_PATH = 'resume.pdf'   # The user's uploaded resume in PDF form
SENTENCE_BERT_MODEL = 'all-MiniLM-L6-v2'  # Sentence-BERT model
CHOSEN_INDUSTRY = "Information-Technology"  
CHOSEN_EXPERIENCE = "junior"
SKILL_MULTIPLIER = 2.0  # Increase skill weight if desired

CATEGORY_SKILLS = {
    'HR': [
        'recruiting', 'employee relations', 'benefits', 'compliance', 
        'training', 'talent acquisition', 'performance management'
    ],
    'Designer': [
        'graphic design', 'ui', 'ux', 'illustration', 'branding', 
        'adobe', 'prototyping', 'typography'
    ],
    'Information-Technology': [
        'network', 'cybersecurity', 'python', 'sql', 
        'cloud', 'help desk', 'project management', 'devops'
    ],
    'Teacher': [
        'curriculum', 'classroom management', 'lesson planning', 
        'assessment', 'differentiated instruction', 'edtech', 'pedagogy'
    ],
    'Advocate': [
        'legal research', 'client advocacy', 'litigation', 
        'policy', 'negotiation', 'case management', 'contract drafting'
    ],
    'Business-Development': [
        'lead generation', 'partnership', 'market research', 
        'sales strategy', 'business strategy', 'negotiation', 'strategic planning'
    ],
    'Healthcare': [
        'patient care', 'medical terminology', 'hipaa', 
        'clinical', 'diagnostics', 'health education', 'care coordination'
    ],
    'Fitness': [
        'personal training', 'exercise', 'nutrition', 
        'wellness', 'fitness assessments', 'group fitness', 'strength training'
    ],
    'Agriculture': [
        'crop management', 'soil', 'irrigation', 
        'farm machinery', 'sustainable farming', 'pest control', 'harvest'
    ],
    'BPO': [
        'customer service', 'call center', 'data entry', 
        'outsourcing', 'telemarketing', 'crm', 'sla management'
    ],
    'Sales': [
        'prospecting', 'lead nurturing', 'cold calling', 'closing deals', 
        'account management', 'salesforce', 'pipeline management'
    ],
    'Consultant': [
        'strategic planning', 'market analysis', 'process improvement', 
        'stakeholder engagement', 'data analysis', 'client presentations', 'change management'
    ],
    'Digital-Media': [
        'social media', 'content creation', 'seo', 'sem', 
        'video editing', 'analytics', 'influencer', 'campaign management'
    ],
    'Automobile': [
        'vehicle maintenance', 'automotive engineering', 'diagnostic tools', 
        'car sales', 'supply chain', 'quality control', 'auto repair'
    ],
    'Chef': [
        'menu planning', 'food preparation', 'culinary', 
        'kitchen management', 'food safety', 'nutrition', 'inventory control'
    ],
    'Finance': [
        'financial analysis', 'budgeting', 'accounting', 'forecasting', 
        'investments', 'risk management', 'excel', 'financial modeling'
    ],
    'Apparel': [
        'fashion design', 'textile', 'merchandising', 
        'pattern making', 'quality control', 'trend analysis', 'inventory management'
    ],
    'Engineering': [
        'cad', 'mechanical design', 'electrical engineering', 'system design', 
        'project management', 'quality assurance', 'prototyping'
    ],
    'Accountant': [
        'bookkeeping', 'tax preparation', 'financial reporting', 
        'auditing', 'sap', 'quickbooks', 'gaap', 'reconciliation'
    ],
    'Construction': [
        'blueprint', 'project scheduling', 'safety compliance', 
        'cost estimation', 'material procurement', 'team management', 'budgeting'
    ],
    'Public-Relations': [
        'press releases', 'media relations', 'crisis management', 
        'branding', 'event planning', 'social media', 'stakeholder communication'
    ],
    'Banking': [
        'loan processing', 'credit analysis', 'customer service', 
        'risk assessment', 'investment services', 'compliance', 'financial advisement'
    ],
    'Arts': [
        'artistic', 'art history', 'painting', 
        'sculpting', 'graphic design', 'photography', 'visual storytelling'
    ],
    'Aviation': [
        'flight operations', 'aircraft maintenance', 'air traffic control', 
        'aerodynamics', 'safety regulations', 'logistics', 'flight planning'
    ]
}

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("project_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy 'en_core_web_sm' model successfully.")
except OSError:
    logger.info("spaCy model 'en_core_web_sm' not found. Downloading...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    logger.info("Downloaded and loaded spaCy 'en_core_web_sm' model.")

def check_file_exists(file_path: str) -> bool:
    exists = os.path.exists(file_path)
    if not exists:
        logger.error(f"File not found: {file_path}")
    return exists

def clean_and_lemmatize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def extract_text_from_pdf(pdf_path: str) -> str:
    if not check_file_exists(pdf_path):
        return ""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logger.info(f"Extracted text from PDF: {pdf_path}")
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
    return text.strip()

def create_industry_experience_prompt(industry: str, experience_level: str) -> str:
    return f"Looking for a candidate in the {industry} industry with {experience_level} level experience."

def count_skills_in_text(text: str, skills: list) -> int:
    text_lower = text.lower() if isinstance(text, str) else ""
    count = 0
    for skill in skills:
        count += text_lower.count(skill.lower())
    return count

def extract_years_of_experience(text: str) -> int:
    if not isinstance(text, str):
        return 0
    matches = re.findall(r'(\d+)\s+years', text.lower())
    if matches:
        return max(int(m) for m in matches)
    return 0

def categorize_experience_years(years: int) -> str:
    if years < 3:
        return 'junior'
    elif years < 8:
        return 'mid'
    elif years < 15:
        return 'senior'
    else:
        return 'executive'

def experience_to_numeric(level: str) -> int:
    mapping = {'junior': 1, 'mid': 2, 'senior': 3, 'executive': 4}
    return mapping.get(level, 1)

def main():
    if not check_file_exists(CSV_PATH):
        logger.error("Exiting program due to missing CSV file.")
        return

    try:
        data = pd.read_csv(CSV_PATH)
        logger.info("Loaded data from CSV successfully.")
    except Exception as e:
        logger.error(f"Error loading CSV file {CSV_PATH}: {e}")
        return

    required_columns = ['ID', 'Resume_str', 'Resume_html', 'Category']
    for col in required_columns:
        if col not in data.columns:
            logger.error(f"CSV file must contain a '{col}' column.")
            return

    # Data Cleaning
    logger.info("Preprocessing resumes from dataset...")
    data['Cleaned_Resume'] = data['Resume_str'].apply(clean_and_lemmatize)

    # Load Sentence-BERT model
    try:
        logger.info("Loading Sentence-BERT model...")
        model = SentenceTransformer(SENTENCE_BERT_MODEL)
    except Exception as e:
        logger.error(f"Error loading Sentence-BERT model '{SENTENCE_BERT_MODEL}': {e}")
        return

    # Encode dataset resumes
    try:
        logger.info("Encoding dataset resumes with Sentence-BERT...")
        dataset_embeddings = model.encode(
            data['Cleaned_Resume'].tolist(),
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=True
        )
    except Exception as e:
        logger.error(f"Error during dataset resume embedding: {e}")
        return

    # User Resume
    user_resume_text = extract_text_from_pdf(UPLOADED_PDF_PATH)
    user_cleaned_resume = clean_and_lemmatize(user_resume_text)
    user_embedding = model.encode([user_cleaned_resume], convert_to_tensor=True)

    query_text = create_industry_experience_prompt(CHOSEN_INDUSTRY, CHOSEN_EXPERIENCE)
    query_embedding = model.encode([query_text], convert_to_tensor=True)

    user_score = util.cos_sim(user_embedding, query_embedding).item()
    logger.info(f"User Resume Suitability Score for {CHOSEN_INDUSTRY} ({CHOSEN_EXPERIENCE} level): {user_score:.4f}")

    all_scores = util.cos_sim(user_embedding, dataset_embeddings).cpu().numpy().flatten()
    data['User_Similarity'] = all_scores

    # Skill-Based Features
    chosen_skills = CATEGORY_SKILLS.get(CHOSEN_INDUSTRY, [])
    data['Skill_Count'] = data['Resume_str'].apply(lambda x: count_skills_in_text(x, chosen_skills))

    # Experience Extraction
    data['Years_Exp'] = data['Resume_str'].apply(extract_years_of_experience)
    data['Experience_Level'] = data['Years_Exp'].apply(categorize_experience_years)
    data['Experience_Num'] = data['Experience_Level'].apply(experience_to_numeric)

    user_years_exp = extract_years_of_experience(user_resume_text)
    user_exp_level = categorize_experience_years(user_years_exp)
    user_exp_num = experience_to_numeric(user_exp_level)
    logger.info(f"User Resume Experience Level: {user_exp_level} ({user_years_exp} years)")

    # Statistical Test (Engineering vs Finance)
    eng_scores = data.loc[data['Category'].str.lower() == 'engineering', 'User_Similarity']
    fin_scores = data.loc[data['Category'].str.lower() == 'finance', 'User_Similarity']
    if len(eng_scores) > 1 and len(fin_scores) > 1:
        t_stat, p_val = ttest_ind(eng_scores, fin_scores, equal_var=False, nan_policy='omit')
        logger.info(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
        if p_val < 0.05:
            logger.info("We reject the null hypothesis and conclude there's a significant difference.")
        else:
            logger.info("We fail to reject the null hypothesis; no significant difference found.")
    else:
        logger.info("Not enough data to perform the statistical test.")

    # Correlation
    category_to_num = {cat: i for i, cat in enumerate(sorted(data['Category'].unique()))}
    data['Category_Num'] = data['Category'].map(category_to_num)
    corr = data['User_Similarity'].corr(data['Category_Num'])
    logger.info(f"Correlation between category numeric encoding and User_Similarity: {corr:.4f}")

    # Predictive Model
    data['Good_Fit'] = data['Category'].apply(lambda c: 1 if c.lower() == CHOSEN_INDUSTRY.lower() else 0)

    # Weighted skill features
    data['Weighted_Skill'] = data['Skill_Count'] * SKILL_MULTIPLIER

    # Features: User_Similarity, Weighted_Skill, Experience_Num
    X = data[['User_Similarity', 'Weighted_Skill', 'Experience_Num']].fillna(0)
    y = data['Good_Fit']

    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Handle imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Logistic Regression with class_weight
    model_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_lr.fit(X_train_res, y_train_res)

    y_pred_proba = model_lr.predict_proba(X_test)[:, 1]

    # Threshold Tuning
    thresholds = np.linspace(0, 1, 101)
    best_thresh = 0.5
    best_f1 = 0
    for t in thresholds:
        y_pred_adj = (y_pred_proba > t).astype(int)
        f1_pos = f1_score(y_test, y_pred_adj, pos_label=1)
        if f1_pos > best_f1:
            best_f1 = f1_pos
            best_thresh = t

    logger.info(f"Best threshold found: {best_thresh:.2f} with F1-score for positive class: {best_f1:.4f}")
    y_pred = (y_pred_proba > best_thresh).astype(int)

    logger.info("Classification Report after Threshold Tuning:")
    logger.info("\n" + classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"Model AUC: {auc:.4f}")

    # Data Visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Category', y='User_Similarity', data=data)
    plt.xticks(rotation=90)
    plt.title('User Similarity Scores by Category')
    plt.tight_layout()
    plt.savefig("category_similarity_boxplot.png")
    plt.close()

    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Good_Fit Prediction')
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    logger.info("Data visualization saved as 'category_similarity_boxplot.png' and 'roc_curve.png'.")

if __name__ == "__main__":
    main()
