from flask import Flask, request, jsonify, render_template, send_file
import spacy
import re
from pdfminer.high_level import extract_text
import os
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from groq import Groq

app = Flask(__name__, template_folder="templates", static_folder="static")

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    text = extract_text(file_path)
    text = re.sub(r'\n\s*\n+', '\n', text).strip()
    print(f"Raw Extracted Text: {text}")
    return text

def extract_contact_info(text):
    email = re.search(r'[\w\.-]+@[\w\.-]+', text)
    phone = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
    
    name = None
    lines = text.split('\n')
    for i, line in enumerate(lines[:5]):
        line = line.strip()
        if re.match(r'^[A-Z][a-zA-Z]* [A-Z][a-zA-Z]*$', line):
            name = line
            break
        if re.match(r'^[A-Z]+ [A-Z]+$', line):
            name = line.title()
            break
        if i < len(lines) - 1:
            
            email_in_next_line = re.search(r'[\w\.-]+@[\w\.-]+', lines[i + 1]) if email else None
            phone_in_next_line = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', lines[i + 1]) if phone else None
            if (email_in_next_line or phone_in_next_line) and re.match(r'^[A-Za-z]+ [A-Za-z]+$', line):
                name = line.title()
                break
    
    return {
        "name": name if name else "Your Name",
        "email": email.group() if email else "email@example.com",
        "phone": phone.group() if phone else "123-456-7890"
    }

def save_as_pdf(text, filename, name="Your Name", email="email@example.com", phone="123-456-7890"):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    name_style = ParagraphStyle(
        'Name', parent=styles['Heading1'], fontSize=24, textColor=colors.black, spaceAfter=6, alignment=1, fontName='Helvetica-Bold'
    )
    cv_style = ParagraphStyle(
        'CV', parent=styles['Heading2'], fontSize=16, textColor=colors.black, spaceAfter=24, alignment=1, fontName='Helvetica-Bold'
    )
    section_style = ParagraphStyle(
        'Section', parent=styles['Heading3'], fontSize=14, textColor=colors.black, spaceAfter=6, spaceBefore=12, fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.black, spaceAfter=4, leftIndent=20  # Add left indent for bullet points
    )

    story = []
    story.append(Paragraph(name, name_style))
    story.append(Paragraph("Curriculum Vitae", cv_style))

    sections = {
        "Personal Details": [], "Education": [], "Work Experience": [], "Skills": [],
        "Tech Skills": [], "Soft Skills": [], "Projects": [], "Achievements": [], "Languages": [], "Contact": []
    }
    current_section = None
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^(Personal Details|Education|Work Experience|Skills|Tech Skills|Soft Skills|Projects|Achievements|Languages|Contact)$', line, re.IGNORECASE):
            current_section = line.title()
        elif current_section:
            sections[current_section].append(line)
        elif re.search(r'[\w\.-]+@[\w\.-]+|\d{3}[-.]?\d{3}[-.]?\d{4}', line):
            sections["Personal Details"].append(line)
        else:
            sections["Projects"].append(line)

    
    for section, content in sections.items():
        if content:
          
            story.append(Paragraph(section, section_style))
            
           
            for item in content:
               
                if item.startswith('•') or item.startswith('-'):
                    bullet_text = item
                else:
                    bullet_text = f"• {item}"
                story.append(Paragraph(bullet_text, body_style))
            
            
            story.append(Spacer(1, 12))

    doc.build(story)
    print(f"PDF Generated: {filename}")
    return filename

def correct_grammar(text, job_description="", missing_keywords=None):
    print(f"Original Text for Enhancement: {text}")
    prompt = """You are an expert resume editor and career coach specializing in creating highly effective, results-driven resumes that capture the attention of hiring managers. Enhance the entire resume text below by following these strict guidelines:
1. Correct all grammar, spelling, and punctuation errors across the entire text to ensure flawless language.
2. Rewrite every sentence and bullet point to use dynamic, results-oriented language that highlights achievements, leadership, and innovation, using sophisticated vocabulary where appropriate.
3. Convert passive voice to active voice in all sentences to emphasize initiative and ownership.
4. Replace weak verbs (e.g., 'did', 'made', 'got', 'used', 'worked', 'helped', 'responsible for', 'handled') with powerful, action-oriented verbs (e.g., 'orchestrated', 'engineered', 'accelerated', 'optimized', 'executed', 'spearheaded', 'delivered', 'forged') in every sentence.
5. Quantify achievements with specific, realistic metrics in every relevant sentence to demonstrate impact (e.g., 'boosted efficiency by 20%', 'slashed costs by 15%', 'launched 10+ projects annually'). If metrics are not provided, infer reasonable numbers based ONLY on the context of the original text—do not invent new achievements or details.
6. Seamlessly incorporate the following missing keywords (if provided) into relevant sections like Work Experience or Skills, ensuring they fit naturally and contextually across the resume: {missing_keywords}. If no keywords are provided, do not add any.
7. Maintain the same overall structure and content, ensuring all sections (e.g., Personal Details, Education, Work Experience, Skills, Tech Skills, Soft Skills, Projects, Achievements, Languages, Contact) and bullet points are preserved with their original headings intact (do not merge, invent new sections, or add content not present in the original text).
8. Process the entire resume text, including all sentences and bullet points, without stopping at full stops or newlines, to ensure every line is enhanced.
9. Ensure all bullet points and sections remain intact for clarity and readability, preserving the original formatting (e.g., newlines between sections, bullet points with '•' or '-').
10. Minimize the use of 'I' by focusing on action verbs and results (e.g., change 'I developed' to 'Developed' or 'Engineered'), enhancing professionalism and conciseness.
11. **Critical Instruction**: Do not add any new information, achievements, or details that are not explicitly present in the original resume text. Only enhance the existing content without expanding beyond what is provided.

Respond ONLY with the improved resume text, with no explanations, comments, or additional formatting.
Follow the 11th point strictly 
"""
    if job_description:
        prompt += "\n\nOptimize the resume to align with this job description without adding new content beyond the original resume:\n" + job_description
    if missing_keywords:
        prompt = prompt.format(missing_keywords=", ".join(missing_keywords))
    else:
        prompt = prompt.format(missing_keywords="None")
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            max_completion_tokens=2000,
            temperature=0.5,
            stream=False
        )
        corrected_text = response.choices[0].message.content.strip()
        print(f"Groq API Response: {corrected_text}")
        return corrected_text
    except Exception as e:
        print(f"Exception when calling Groq API: {e}")
        return None

def extract_keywords(text):
    doc = nlp(text.lower())
    keywords = set()
    for chunk in doc.noun_chunks:
        if not all(token.is_stop for token in chunk):
            keywords.add(chunk.text.strip())
    for token in doc:
        if token.is_alpha and not token.is_stop and len(token.text) > 1:
            if token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB"]:
                keywords.add(token.text.strip())
    return list(keywords)

def suggest_keywords(resume_text, job_description):
    if not job_description.strip():
        return []
    resume_keywords = set(extract_keywords(resume_text))
    job_keywords = set(extract_keywords(job_description))
    important_missing = job_keywords - resume_keywords
    job_desc_lower = job_description.lower()
    sorted_missing = sorted(important_missing, key=lambda x: job_desc_lower.count(x), reverse=True)
    return sorted_missing[:15]

def count_grammar_issues(text):
    issues = 0
    common_errors = [
        r'\b(its|it\'s)\b', r'\b(their|there|they\'re)\b', r'\b(your|you\'re)\b',
        r'\b(affect|effect)\b', r'\b(then|than)\b', r'\ba\s+[aeiouh]', r'\ban\s+[^aeiouh]',
        r'\s{2,}', r'[.!?]\s*[A-Za-z]'
    ]
    for pattern in common_errors:
        issues += len(re.findall(pattern, text))
    lines = text.split('\n')
    sentences = []
    for line in lines:
        if not line.strip() or re.match(r'^(Personal Details|Education|Work Experience|Skills|Tech Skills|Soft Skills|Projects|Achievements|Languages|Contact)$', line.strip(), re.IGNORECASE) or line.strip().startswith('•'):
            continue
        sentences.extend(re.split(r'[.!?]+', line))
    for sentence in sentences:
        word_count = len(sentence.split())
        if 0 < word_count < 3 and sentence.strip():
            issues += 1
        elif word_count > 30 and sentence.strip():
            issues += 1
    total_issues = min(issues, 10)
    print(f"Total grammar issues: {total_issues}")
    return total_issues

def check_formatting(paragraphs):
    formatting_issues = 0
    styles_used = set()
    fonts_used = set()
    sizes_used = set()
    for para in paragraphs:
        styles_used.add(para["style"])
        for run in para["runs"]:
            if run.get("font_name"):
                fonts_used.add(run.get("font_name"))
            if run.get("font_size"):
                sizes_used.add(run.get("font_size"))
    if len(styles_used) > 5:
        formatting_issues += min(len(styles_used) - 5, 5)
    if len(fonts_used) > 2:
        formatting_issues += min(len(fonts_used) - 2, 5)
    if len(sizes_used) > 4:
        formatting_issues += min(len(sizes_used) - 4, 5)
    has_header = any("Heading" in style for style in styles_used)
    if not has_header:
        formatting_issues += 5
    if len(paragraphs) < 5:
        formatting_issues += 3
    elif len(paragraphs) > 50:
        formatting_issues += 3
    formatting_score = max(20 - formatting_issues, 0)
    return formatting_score

def check_structure(text):
    structure_score = 10
    essential_sections = [
        r'\b(personal details|profile|summary|objective)\b', r'\b(experience|work experience|employment)\b',
        r'\b(education|academic|university|college)\b', r'\b(skills|competencies|expertise)\b'
    ]
    sections_found = sum(1 for pattern in essential_sections if re.search(pattern, text, re.IGNORECASE))
    if sections_found < len(essential_sections):
        structure_score -= (len(essential_sections) - sections_found) * 2
    has_contact = re.search(r'\b(email|phone|tel|contact|\d{3}[-.]?\d{3}[-.]?\d{4})\b', text, re.IGNORECASE)
    if not has_contact:
        structure_score -= 2
    bullets = len(re.findall(r'•|\*', text))
    if bullets < 5:
        structure_score -= 1
    return max(structure_score, 0)

def check_action_verbs(text):
    text = text.lower()
    strong_verbs = [
        'achieved', 'managed', 'developed', 'led', 'implemented', 'improved', 'increased', 'designed',
        'coordinated', 'executed', 'analyzed', 'delivered', 'established', 'facilitated', 'generated',
        'orchestrated', 'engineered', 'accelerated', 'optimized', 'spearheaded', 'forged'
    ]
    weak_verbs = [
        'was', 'were', 'did', 'made', 'got', 'used', 'worked', 'helped', 'responsible for', 'handled'
    ]
    strong_count = sum(len(re.findall(r'\b' + verb + r'\b', text)) for verb in strong_verbs)
    weak_count = sum(len(re.findall(r'\b' + verb + r'\b', text)) for verb in weak_verbs)
    total_verbs = strong_count + weak_count
    if total_verbs == 0:
        return []
    weak_verb_suggestions = []
    if weak_count > 0:
        for verb in weak_verbs:
            if re.search(r'\b' + verb + r'\b', text):
                if verb == 'responsible for':
                    weak_verb_suggestions.append("Replace 'responsible for' with 'orchestrated' or 'spearheaded'")
                elif verb == 'worked':
                    weak_verb_suggestions.append("Replace 'worked' with 'executed' or 'delivered'")
                elif verb == 'helped':
                    weak_verb_suggestions.append("Replace 'helped' with 'facilitated' or 'accelerated'")
                else:
                    weak_verb_suggestions.append(f"Replace '{verb}' with stronger action verbs like 'engineered' or 'forged'")
    return weak_verb_suggestions

def calculate_resume_score(resume_text, job_description, original_paragraphs):
    grammar_issues = count_grammar_issues(resume_text)
    grammar_score = max(40 - grammar_issues * 4, 0)
    if job_description.strip():
        job_keywords = extract_keywords(job_description)
        resume_keywords = extract_keywords(resume_text)
        if job_keywords:
            matching_keywords = set(resume_keywords).intersection(set(job_keywords))
            keyword_match_percentage = len(matching_keywords) / len(job_keywords)
            keyword_score = min(int(keyword_match_percentage * 30), 30)
        else:
            keyword_score = 15
    else:
        keyword_score = 15
    formatting_score = check_formatting(original_paragraphs)
    structure_score = check_structure(resume_text)
    total_score = grammar_score + keyword_score + formatting_score + structure_score
    if total_score >= 90:
        feedback = "Your resume is **Excellent**! It is well-optimized for the job description."
    elif total_score >= 80:
        feedback = "Your resume is **Very Good**. Some minor improvements can make it even better."
    elif total_score >= 70:
        feedback = "Your resume is **Good**. Several improvements are recommended."
    elif total_score >= 60:
        feedback = "Your resume is **Fair**. Significant improvements are needed."
    else:
        feedback = "Your resume needs **Major Improvements**. Consider a substantial revision."
    return {
        "grammar_score": grammar_score,
        "keyword_score": keyword_score,
        "formatting_score": formatting_score,
        "structure_score": structure_score,
        "total_score": total_score,
        "feedback": feedback
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/enhance", methods=["POST"])
def enhance():
    resume_file = request.files["resume"]
    job_description = request.form.get("job_description", "")
    temp_file_path = None
    
    try:
        if not os.path.exists("temp"):
            os.makedirs("temp")
        
        unique_id = str(int(time.time())) + "_" + str(hash(resume_file.filename))[:8]

        if resume_file.filename.endswith(".pdf"):
            temp_file_path = os.path.join("temp", resume_file.filename)
            resume_file.save(temp_file_path)
            resume_text = extract_text_from_pdf(temp_file_path)
            print(f"Extracted Resume Text: {resume_text}")
            paragraphs = [{"text": p, "style": "Normal", "runs": [{"text": p, "bold": False, "italic": False}]} 
                          for p in resume_text.split('\n') if p.strip()]
            missing_keywords = suggest_keywords(resume_text, job_description)
            corrected_text = correct_grammar(resume_text, job_description, missing_keywords)
            if corrected_text is None:
                return jsonify({"error": "Failed to enhance resume. Please check your Groq API key or try again later."})
            print(f"Corrected Text: {corrected_text}")
            corrected_file_path_pdf = f"corrected_resume_{unique_id}.pdf"
            contact_info = extract_contact_info(resume_text)
            save_as_pdf(corrected_text, corrected_file_path_pdf, **contact_info)
            download_links = {
                "pdf": f"/download/{corrected_file_path_pdf}"
            }

        else:
            return jsonify({"error": "Unsupported file format. Please upload a PDF file."})

        missing_keywords = suggest_keywords(corrected_text, job_description)
        score_breakdown = calculate_resume_score(corrected_text, job_description, paragraphs)
        action_verb_suggestions = check_action_verbs(resume_text)
        diff = {
            "original_length": len(resume_text),
            "enhanced_length": len(corrected_text),
            "difference_percentage": round((len(corrected_text) - len(resume_text)) / len(resume_text) * 100, 1) if len(resume_text) > 0 else 0
        }

        print(f"Download links: {download_links}")
        return jsonify({
            "corrected_text": corrected_text,
            "original_text": resume_text,
            "missing_keywords": missing_keywords,
            "score_breakdown": score_breakdown,
            "action_verb_suggestions": action_verb_suggestions,
            "download_links": download_links,
            "diff_info": diff
        })
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"An error occurred: {str(e)}"})
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.route("/download/<path:filename>")
def download(filename):
    full_path = os.path.abspath(filename)
    print(f"Serving file: {full_path}")
    if os.path.exists(full_path):
        return send_file(full_path, as_attachment=True)
    else:
        return jsonify({"error": f"File {filename} not found"}), 404

if __name__ == "__main__":
    if not os.path.exists("temp"):
        os.makedirs("temp")
    app.run(debug=True)