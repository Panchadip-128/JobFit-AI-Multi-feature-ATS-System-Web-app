import gradio as gr
import pdfplumber
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import re
import io
import base64

nltk.download('stopwords')

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
stop_words = set(stopwords.words("english"))

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return words

def get_top_keywords(text, n=15):
    words = clean_text(text)
    most_common = Counter(words).most_common(n)
    return [word for word, _ in most_common]

def generate_wordcloud(text, title):
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(clean_text(text)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=18)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def evaluate_resumes(resume_files, job_description, top_n):
    if not resume_files or not job_description.strip():
        return "Please upload resumes and provide a job description.", None, None, []

    jd_text = job_description.strip()
    jd_keywords = get_top_keywords(jd_text)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)

    results = []
    for file in resume_files:
        resume_text = extract_text_from_pdf(file.name)
        resume_keywords = get_top_keywords(resume_text)
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)

        # Semantic similarity score
        similarity_score = float(util.cos_sim(resume_embedding, jd_embedding)[0][0])

        # Keyword overlap score
        overlap = len(set(jd_keywords) & set(resume_keywords)) / len(set(jd_keywords)) if jd_keywords else 0

        # Fit score as weighted average
        fit_score = round((0.7 * similarity_score + 0.3 * overlap) * 100, 2)

        results.append({
            "filename": os.path.basename(file.name),
            "fit_score": fit_score,
            "similarity": round(similarity_score * 100, 2),
            "overlap": round(overlap * 100, 2),
        })

    # Sort by fit score
    results = sorted(results, key=lambda x: x["fit_score"], reverse=True)
    top_results = results[:top_n]

    # Generate WordClouds
    resume_all_text = " ".join([extract_text_from_pdf(file.name) for file in resume_files])
    wc1 = generate_wordcloud(resume_all_text, "Resumes Word Cloud")
    wc2 = generate_wordcloud(jd_text, "Job Description Word Cloud")

    return None, wc1, wc2, top_results

def launch_app():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# JobFit AI: Smart ATS Resume Evaluator")
        gr.Markdown("Upload resumes and a job description to get fit scores, insights, and top resume matches.")

        with gr.Row():
            resumes = gr.File(label="Upload Resume PDFs", file_types=[".pdf"], file_count="multiple")
            jd_input = gr.Textbox(label="Paste Job Description", lines=10, placeholder="Enter job description here...")

        top_n_slider = gr.Slider(label="Top N Resumes to Display", minimum=1, maximum=10, value=3, step=1)

        submit_btn = gr.Button("Evaluate Resumes", variant="primary")

        error_output = gr.Textbox(label="Status", visible=False)
        with gr.Row():
            resume_wc = gr.Image(label="Resumes Word Cloud")
            jd_wc = gr.Image(label="JD Word Cloud")

        result_table = gr.Dataframe(headers=["Filename", "Fit Score", "Similarity %", "Keyword Match %"], label="Top Matching Resumes")

        submit_btn.click(
            evaluate_resumes,
            inputs=[resumes, jd_input, top_n_slider],
            outputs=[error_output, resume_wc, jd_wc, result_table]
        )

    demo.launch()

if __name__ == "__main__":
    launch_app()
