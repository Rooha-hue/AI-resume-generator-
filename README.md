# AI-resume-generator-
# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get an **ATS score** plus practical suggestions to improve it.
Built with **Streamlit** (UI) and **Google Gemini Flash** (AI analysis).

## Features

- Upload PDF, DOCX or TXT resumes (max 5 MB)
- Optional job description for keyword matching
- Overall ATS score (0–100) with a breakdown: keywords, formatting, content quality, relevance
- Strengths, prioritised improvements, missing keywords
- Before/after examples for weak bullet points
- Quick rule-based checks: contact info, key sections, length, bullets, quantified achievements
- Shows the exact text an ATS can read from your file
- Still works (rule-based score only) if the AI call fails

## How the score is calculated

`Overall = 70% AI score + 30% rule-based score`

- **AI score** – Gemini evaluates the resume against ATS best practices (and the job description if given).
- **Rule-based score** – deterministic checks worth 100 points: email (10), phone (5), links (5),
  Experience (15), Education (10), Skills (15), Summary (5), length (10), bullets (10), quantified achievements (15).

> This is an estimate. Real ATS products (Workday, Greenhouse, Lever…) each behave differently,
> so treat the score as guidance, not a guarantee.

## Project structure

```
ats-resume-checker/
├── app.py            # the whole app
├── requirements.txt
└── README.md
```

## Run locally

1. Install Python 3.10+.
2. Get a free Gemini API key: https://aistudio.google.com/apikey
3. Set up the project:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Add your key (pick one):

```bash
# Option A: Streamlit secrets (recommended)
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "your-key-here"' > .streamlit/secrets.toml

# Option B: environment variable
export GEMINI_API_KEY="your-key-here"       # Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
```

If no key is found, the app shows a password box in the sidebar.

5. Start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this project to a **GitHub** repository (never commit `.streamlit/secrets.toml`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, and main file path `app.py`.
4. Open **Advanced settings → Secrets** and paste:

```toml
GEMINI_API_KEY = "your-key-here"
```

5. Click **Deploy**. Every `git push` to the branch redeploys automatically.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | Streamlit secrets / env var / sidebar | – |
| `GEMINI_MODEL` | env var or secret exposed as env var | `gemini-2.5-flash` |

## Privacy

Resume text is sent to the Gemini API for analysis. The app does not save uploaded files.
Check Google's data-use terms for your API tier before using real candidates' data.

## Limitations

- Scanned / image-only PDFs have no readable text and are rejected with a clear message.
- Very long resumes are truncated to 15,000 characters before analysis.
- AI output can vary slightly between runs.
