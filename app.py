import gradio as gr
import pdfplumber
import re
import io
import matplotlib.pyplot as plt
import tempfile
from sentence_transformers import SentenceTransformer, util
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Download NLTK resources
nltk.download('stopwords')

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return ""
    return text.strip()

# Clean text
def clean_text(text):
    return re.sub(r'[^\w\s.,;:!?()-]', '', text)

# Extract keywords
def extract_keywords(text, top_n=20):
    words = re.findall(r'\b\w{4,}\b', text.lower())
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    filtered = [word for word in words if word not in stop_words]
    freq = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1
    sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_kw[:top_n])

# Generate WordCloud
def generate_wordcloud(freq_dict):
    if not freq_dict:
        return None
    wc = WordCloud(width=600, height=400, background_color="white")
    wc.generate_from_frequencies(freq_dict)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        plt.figure(figsize=(6, 4))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(tmp.name, format="png")
        return tmp.name

# Evaluate a resume
def evaluate_resume(resume_text, job_desc):
    if not resume_text:
        return 0.0, {}, {}

    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(job_desc, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(resume_emb, jd_emb).item()

    resume_kw = extract_keywords(resume_text)
    jd_kw = extract_keywords(job_desc)
    overlap = len(set(resume_kw.keys()) & set(jd_kw.keys()))
    keyword_score = (overlap / len(jd_kw)) if jd_kw else 0

    format_score = 1.0 if len(resume_text.split()) >= 100 else 0.5

    final_score = (0.5 * similarity + 0.3 * keyword_score + 0.2 * format_score) * 100
    return round(final_score, 2), resume_kw, jd_kw

# ATS evaluation
def ats_batch_checker(resume_files, job_desc, top_n):
    job_desc = clean_text(job_desc)
    if not job_desc.strip():
        return "❌ Please provide a valid job description.", None, None, None, []

    results = []
    all_resume_kw = {}
    jd_kw_freq = extract_keywords(job_desc)

    for file in resume_files:
        resume_text = extract_text_from_pdf(file.name)
        score, resume_kw, jd_kw = evaluate_resume(resume_text, job_desc)
        all_resume_kw[file.name] = resume_kw
        results.append((file.name.split('/')[-1], score, resume_kw))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    top_results = results[:top_n]

    table = "### Top Matching Resumes\n\n| Resume | ATS Score (%) |\n|--------|----------------|\n"
    for name, score, _ in top_results:
        table += f"| {name} | {score} |\n"

    first_resume_kw = top_results[0][2] if top_results else {}
    resume_wc = generate_wordcloud(first_resume_kw)
    jd_wc = generate_wordcloud(jd_kw_freq)

    return table, "✅ Evaluation Complete", resume_wc, jd_wc, top_results

# Send interview invitation email
def send_email(candidate_email, job_desc):
    sender_email = "panchadip125@gmail.com"       # Replace with your email
    password = "fqgs xyxy yxyxy xyxy"              # Use app-specific password for Gmail or SMTP, currently its a dummy value, you can find it under your google account -> Less Secure apps/ Set app passwords

    subject = "Interview Invitation"
    body = f"""
    Dear Candidate,

    Congratulations! Your resume has been shortlisted for the next round.

    Here is the job description:
    {job_desc}

    Please reply to confirm your availability for the interview.

    Best regards,
    Hiring Team
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = candidate_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, candidate_email, msg.as_string())
        return "✅ Interview invitation sent successfully!"
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("## 🤖 AI Resume ATS Evaluator")
    gr.Markdown("🚀 Upload resumes, evaluate them using AI, and invite top candidates for an interview.")

    with gr.Row():
        resume_input = gr.File(label="📄 Upload Resume PDFs", file_types=[".pdf"], file_count="multiple")
        top_n_slider = gr.Slider(minimum=1, maximum=10, value=3, label="🎯 Top N Candidates")

    jd_input = gr.Textbox(label="📌 Job Description", lines=8, placeholder="Paste the job description here...")

    submit_btn = gr.Button("💡 Evaluate Resumes")

    with gr.Row():
        ats_summary = gr.Markdown(label="📊 ATS Result Summary")
        status_text = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        resume_wc = gr.Image(label="🔍 Resume WordCloud")
        jd_wc = gr.Image(label="🧠 Job Description WordCloud")

    email_input = gr.Textbox(label="📧 Candidate Email", placeholder="Enter candidate's email to send invitation")
    email_btn = gr.Button("📨 Send Interview Email")

    # Logic connections
    submit_btn.click(
    fn=ats_batch_checker,
    inputs=[resume_input, jd_input, top_n_slider],
    outputs=[ats_summary, status_text, resume_wc, jd_wc]  # Removed None
)

    email_btn.click(
        fn=send_email,
        inputs=[email_input, jd_input],
        outputs=[status_text]
    )

iface.launch(debug=True, share=True)
