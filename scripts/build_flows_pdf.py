"""Build docs/VoiceSense-app-flows.pdf.

Run from the project folder:

    python scripts/build_flows_pdf.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "VoiceSense-app-flows.pdf"

pdfmetrics.registerFont(TTFont("Body", r"C:\Windows\Fonts\segoeui.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", r"C:\Windows\Fonts\segoeuib.ttf"))
pdfmetrics.registerFont(TTFont("Body-Italic", r"C:\Windows\Fonts\segoeuii.ttf"))

INK = colors.HexColor("#1a2332")
MUTED = colors.HexColor("#3d4c63")
NAVY = colors.HexColor("#0f2744")
TEAL = colors.HexColor("#0f5f73")
LINE = colors.HexColor("#d5deea")
ZEBRA = colors.HexColor("#f4f7fb")
AMBER = colors.HexColor("#fff6e8")
GREEN = colors.HexColor("#eef8f3")

PAGE_W, PAGE_H = letter
CONTENT_W = PAGE_W - 1.4 * inch


def styles() -> dict:
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        "CoverKicker", fontName="Body-Bold", fontSize=10, textColor=TEAL,
        tracking=1.2, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        "CoverTitle", fontName="Body-Bold", fontSize=26, leading=32,
        textColor=NAVY, spaceAfter=10,
    ))
    base.add(ParagraphStyle(
        "CoverSub", fontName="Body", fontSize=12, leading=17, textColor=MUTED, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        "H1", fontName="Body-Bold", fontSize=16, leading=20, textColor=NAVY,
        spaceBefore=8, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        "H2", fontName="Body-Bold", fontSize=13, leading=17, textColor=TEAL,
        spaceBefore=10, spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        "Body", fontName="Body", fontSize=10.5, leading=15, textColor=INK, spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        "Small", fontName="Body", fontSize=9, leading=12.5, textColor=MUTED, spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        "Cell", fontName="Body", fontSize=9, leading=12, textColor=INK, alignment=TA_LEFT,
    ))
    base.add(ParagraphStyle(
        "CellHead", fontName="Body-Bold", fontSize=9, leading=12, textColor=colors.white,
    ))
    base.add(ParagraphStyle(
        "Callout", fontName="Body", fontSize=10.5, leading=15, textColor=INK,
    ))
    base.add(ParagraphStyle(
        "FooterNote", fontName="Body", fontSize=8, textColor=MUTED,
    ))
    return base


S = styles()


def P(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, S[style])


def h1(text: str) -> Paragraph:
    return P(text, "H1")


def h2(text: str) -> Paragraph:
    return P(text, "H2")


def cell(text: str, head: bool = False) -> Paragraph:
    return Paragraph(text, S["CellHead"] if head else S["Cell"])


def table(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    data = [[cell(h, head=True) for h in headers]]
    for row in rows:
        data.append([cell(c) for c in row])
    grid = Table(data, colWidths=widths, repeatRows=1)
    grid.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEBRA]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
    ]))
    return grid


def callout(text: str, fill=AMBER) -> Table:
    box = Table([[P(text, "Callout")]], colWidths=[CONTENT_W])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return box


def bullets(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(P(item), leftIndent=12, bulletColor=TEAL) for item in items],
        bulletType="bullet",
        start="•",
        leftIndent=16,
        bulletFontName="Body",
        bulletFontSize=10,
        spaceBefore=2,
        spaceAfter=8,
    )


def paint_later(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 32, PAGE_W, 32, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Body", 9)
    canvas.drawString(0.7 * inch, PAGE_H - 20, "VoiceSense   ·   how the app works")
    canvas.setFillColor(MUTED)
    canvas.setFont("Body", 8)
    canvas.drawString(0.7 * inch, 0.42 * inch, "Research screening only. Not a medical diagnosis.")
    canvas.drawRightString(PAGE_W - 0.7 * inch, 0.42 * inch, str(doc.page))
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(0.7 * inch, 0.58 * inch, PAGE_W - 0.7 * inch, 0.58 * inch)
    canvas.restoreState()


def paint_first(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 10, PAGE_W, 10, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Body", 8)
    canvas.drawString(0.7 * inch, 0.42 * inch, "Research screening only. Not a medical diagnosis.")
    canvas.drawRightString(PAGE_W - 0.7 * inch, 0.42 * inch, str(doc.page))
    canvas.restoreState()


def story() -> list:
    w = CONTENT_W
    flow: list = []
    flow += [
        Spacer(1, 28),
        P("VOICESENSE", "CoverKicker"),
        P("How the app works", "CoverTitle"),
        P(
            "A plain guide to every screen, every click, and which source file does the work. "
            "Written so someone new to the project can follow a recording from the microphone to the result.",
            "CoverSub",
        ),
        Spacer(1, 8),
        callout(
            "<b>This is not a medical device.</b> A result is a research screening hint. "
            "It does not diagnose Parkinson's disease, and it does not clear someone of it.",
            AMBER,
        ),
        Spacer(1, 14),
        h1("1. What you are looking at"),
        P(
            "VoiceSense is a local website. You start it on your own computer with "
            "<b>python -m streamlit run app/streamlit_app.py</b>, then open "
            "http://127.0.0.1:8501. The model runs in that Python process. "
            "A successful analysis drops the audio from the page."
        ),
        P("The site has four sections. The control under the title remembers which one you picked, including after you upload a file."),
        table(
            ["Section", "What it is for", "Open this file"],
            [
                ["Live voice studio", "Record or upload a vowel and get a screening hint.", "app/live_page.py"],
                ["Feature lab", "Score a row of numbers from a spreadsheet. No microphone.", "app/feature_page.py"],
                ["Privacy", "See whether the recording stays on this machine, and download a receipt with no WAV.", "app/privacy_ui.py"],
                ["About &amp; method", "Short limits of the method, and where the code lives.", "app/streamlit_app.py"],
            ],
            [1.7 * inch, 3.3 * inch, 2.1 * inch],
        ),
        Spacer(1, 8),
        h2("Folder map"),
        P("Start at the top of this list. Each folder has one job."),
        table(
            ["Folder or file", "Job"],
            [
                ["app/streamlit_app.py", "Website frame only. It draws the title and opens one of the screens."],
                ["app/live_page.py", "Record, upload, discard, analyze, and show the result."],
                ["app/feature_page.py", "Demo row, CSV upload, or typed numbers."],
                ["app/widgets.py", "Shared result cards. Change the look here, not the math."],
                ["app/privacy_ui.py", "On-device banner and receipt download."],
                ["src/__init__.py", "One-line map of the engine. Read this before the other src files."],
                ["src/predict_audio.py", "Score one recording and apply the safety rules."],
                ["src/audio_features.py", "Turn audio into pitch, loudness, and noise numbers."],
                ["src/plain_language.py", "Everyday sentences. Does not change the score."],
                ["src/explain.py", "Technical reasons, compared with training averages."],
                ["src/predict.py and src/train.py", "Spreadsheet models (English UCI, Bengali BenSParX)."],
                ["src/train_audio.py", "Trains the microphone model. Not used when someone clicks Analyze."],
                ["src/privacy.py", "Wipe audio from the page and build a receipt with no WAV."],
                ["models/", "Trained files. Do not edit them by hand."],
                ["reports/", "Accuracy numbers written the last time the models were trained."],
                ["scripts/", "Small checks, including this PDF builder."],
                ["landing/", "A static marketing page. The model does not run there."],
                ["datasets/ (outside this folder)", "Training audio and tables. Not shipped in the git repo."],
            ],
            [2.35 * inch, 4.75 * inch],
        ),
    ]

    flow += [
        CondPageBreak(3.2 * inch),
        h1("2. What happens when you open the site"),
        P("Nothing is scored just because the page loaded. The steps are only setup."),
        table(
            ["Order", "What happens", "File"],
            [
                ["1", "Streamlit loads the entry file and adds the project folder to the Python path.", "app/streamlit_app.py"],
                ["2", "Any leftover audio keys are removed from the session.", "src/privacy.py"],
                ["3", "The title, the research disclaimer, and the privacy banner are drawn.", "app/streamlit_app.py"],
                ["4", "The section control opens Live voice studio, unless you already picked another section in this visit.", "app/streamlit_app.py"],
            ],
            [0.7 * inch, 4.3 * inch, 2.1 * inch],
        ),
        Spacer(1, 6),
        callout(
            "<b>Local versus hosted.</b> On your own machine the banner says on-device. "
            "If the same app is hosted on Streamlit Community Cloud, the browser must send the audio to that server. "
            "The banner says so. That hosted run is not on-device privacy.",
            ZEBRA,
        ),
    ]

    flow += [
        CondPageBreak(3.4 * inch),
        h1("3. Live voice — from a click to a result"),
        P(
            "This is the main path. The person holds a steady vowel, about 5 to 8 seconds, "
            "in a quiet room. English or Bangla only changes the wording and which extra lab score is shown. "
            "The main decision always uses the same microphone model."
        ),
        h2("3.1 You record or upload"),
        P("Uploading or finishing a recording does <b>not</b> score anything, and it does not send you back to the title."),
        table(
            ["You", "The app", "File"],
            [
                ["Choose English or Bangla.", "Stores the choice for this analysis only.", "app/live_page.py"],
                ["Leave preview off.", "The browser does not play the clip, so download managers such as IDM are less likely to grab it.", "app/live_page.py"],
                ["Record, or upload WAV, MP3, OGG, WEBM, M4A, or FLAC.", "Holds the bytes in the widget. Shows how large the upload is. Does not keep the file name in the session.", "app/live_page.py"],
                ["Click Discard recording.", "Clears the widget. Nothing is scored and nothing is saved.", "app/live_page.py"],
            ],
            [2.15 * inch, 3.15 * inch, 1.8 * inch],
        ),
        h2("3.2 You click Analyze voice"),
        table(
            ["Order", "What happens", "File"],
            [
                ["1", "The clip is turned into a WAV in memory. A tiny temp file may be used so the decoder can read WebM or MP4, then that file is deleted.", "app/widgets.py"],
                ["2", "If decoding fails, the original bytes are used instead.", "app/widgets.py"],
                ["3", "Pitch, loudness, noise, and a quality report are measured. Very short, quiet, or noisy clips stop here.", "src/audio_features.py"],
                ["4", "The random-forest microphone model scores the numbers.", "src/predict_audio.py"],
                ["5", "The SVM microphone model scores the same numbers.", "src/predict_audio.py"],
                ["6", "A firm call is kept only when both models agree and the clip is long, clear, and mostly voiced. Otherwise the result is Uncertain.", "src/predict_audio.py"],
                ["7", "Optional extra scores are computed for the English and Bengali lab models. They do not change the main result.", "src/predict_audio.py"],
                ["8", "Everyday sentences are written from the result. They do not change the number.", "src/plain_language.py"],
                ["9", "The recording widget is cleared. The score stays on screen. The audio bytes do not.", "app/live_page.py"],
                ["10", "This visit keeps the latest three scores so a single odd clip is not the whole story.", "src/plain_language.py"],
            ],
            [0.7 * inch, 4.4 * inch, 2.0 * inch],
        ),
        Spacer(1, 6),
        callout(
            "<b>If the clip is rejected.</b> You see a plain sentence and the technical quality numbers inside an expander. "
            "There is no Parkinson's percentage, because the measures would not mean anything.",
            AMBER,
        ),
    ]

    flow += [
        CondPageBreak(3.2 * inch),
        h1("4. How a firm call is allowed"),
        P(
            "The microphone model was trained on studio-style Italian vowels. A laptop microphone "
            "changes pitch-wobble and loudness numbers even when the voice is healthy. "
            "The app would rather say Uncertain than show a false alarm."
        ),
        table(
            ["What you see", "The rule"],
            [
                ["Possible Parkinson's-like pattern", "Main score is 72% or higher, the second model agrees, and the clip passes the checks below."],
                ["No clear pattern", "Main score is 42% or lower, the second model agrees, and the clip passes the same checks."],
                ["Uncertain", "The score is in the middle, the two models disagree, or the clip is too weak to trust."],
            ],
            [2.3 * inch, 4.8 * inch],
        ),
        Spacer(1, 8),
        P("A clip is too weak for a firm call when any of these is true:"),
        bullets([
            "Shorter than 4 seconds.",
            "Background noise is high enough to distort pitch and loudness.",
            "Too little of the clip is a clear voiced sound.",
            "Some voice measures could not be calculated and had to be filled in.",
        ]),
        P(
            "Hard rejects happen even earlier, inside the feature extractor: under 1 second, "
            "very little voiced sound, or a very low signal. Those clips are not scored at all."
        ),
        h2("What the screen shows after a score"),
        bullets([
            "<b>Primary screening decision</b> — the short banner.",
            "<b>In everyday language</b> — what the result means, what the voice showed, how much to trust it, and what to do next. No feature codes.",
            "<b>Your set of recordings</b> — 1 of 3, or a combined reading when several clean takes agree.",
            "<b>Technical details</b> — percentages, quality, the second model, train-time metrics, and the feature table. This block is still there for people who want the numbers.",
        ]),
        callout(
            "<b>Three takes.</b> One home recording is easy to sway. Record the vowel three times, with a pause between them. "
            "A Parkinson's-like reading across the set is shown only when at least two clean takes agree and none of the clean takes points the other way.",
            GREEN,
        ),
    ]

    flow += [
        CondPageBreak(3.0 * inch),
        h1("5. Feature lab"),
        P(
            "This screen never opens a microphone. It compares a row of acoustic numbers with an older research table. "
            "A match is not a diagnosis. Use it to check the lab models, not to screen a person in the room."
        ),
        table(
            ["You pick", "What happens", "File"],
            [
                ["Dataset and classifier", "Loads the English UCI model or the Bengali BenSParX model, random forest or SVM.", "src/predict.py"],
                ["Demo hold-out", "Shows one row that was kept out of training, then you can score it and compare with the true label.", "models/*holdout_samples.csv"],
                ["CSV upload", "Reads your file. You stay on this screen. Missing columns are listed. A matching row can be scored.", "app/feature_page.py"],
                ["Manual", "You type the most important features. The rest can be filled with the training medians.", "app/feature_page.py"],
                ["Run feature screening", "One score, an everyday card, then the technical explanation.", "src/predict.py and src/plain_language.py"],
            ],
            [1.7 * inch, 3.5 * inch, 1.9 * inch],
        ),
        Spacer(1, 6),
        P(
            "The lab models use a yes-or-no cutoff near 50%. The live microphone screen is stricter: "
            "it leaves a wide middle band and can refuse a firm call."
        ),
    ]

    flow += [
        CondPageBreak(3.0 * inch),
        h1("6. Privacy and the receipt"),
        h2("During a live analysis"),
        bullets([
            "Audio is scored in the Python process on the machine running the app.",
            "After a successful analysis the microphone and upload widgets are recreated empty, so the bytes are dropped.",
            "Discard does the same without scoring.",
            "The session does not keep the upload file name.",
        ]),
        h2("Download screening receipt"),
        P(
            "The JSON file is a summary: decision, everyday sentences, and quality. "
            "It has no WAV and no upload file name. The builder checks that audio markers are absent. "
            "Code: <b>src/privacy.py</b>, button in <b>app/privacy_ui.py</b>."
        ),
        P(
            "Identity proofs are not in this repository. The Primus build is the separate project "
            "<b>voicesense-primus</b>."
        ),
    ]

    flow += [
        CondPageBreak(3.0 * inch),
        h1("7. Training is not part of a click"),
        P(
            "Analyze never retrains. It loads files that are already in <b>models/</b>. "
            "Training is a separate command, and it needs datasets that sit next to this project, not inside git."
        ),
        table(
            ["Command", "What it trains", "Needs"],
            [
                ["python -m src.train", "English UCI and Bengali BenSParX spreadsheet models.", "CSV tables under datasets/"],
                ["python -m src.train_audio", "The microphone model used by Live voice studio.", "Italian vowel WAVs under datasets/"],
            ],
            [2.15 * inch, 2.7 * inch, 2.25 * inch],
        ),
        Spacer(1, 8),
        P("Both trainers keep every recording from one person on only one side of the split, so a person is not in both the practice set and the test set. Reports land in <b>reports/</b>."),
        P("The microphone model keeps classical voice measures and leaves out spectral features that memorized the Italian studio microphone and then flagged ordinary laptop mics."),
        h2("Checks you can run without a clinic"),
        bullets([
            "<b>python scripts/check_plain_policy.py</b> — a short clip or a model disagreement cannot become a firm call, and the everyday text hides feature codes.",
            "<b>python scripts/check_privacy.py</b> — a receipt has no audio and no upload name.",
            "<b>python scripts/check_audio_model.py</b> — the saved microphone model file opens.",
        ]),
    ]

    flow += [
        CondPageBreak(3.2 * inch),
        h1("8. I want to change something"),
        table(
            ["Goal", "Open"],
            [
                ["Change the words under the decision banner", "src/plain_language.py"],
                ["Change when a result is allowed to say Parkinson's-like or healthy", "src/predict_audio.py"],
                ["Change how pitch and loudness are measured", "src/audio_features.py"],
                ["Change the technical why-PD list", "src/explain.py and the glossary in src/config.py"],
                ["Change buttons, upload types, or the three-recording note", "app/live_page.py"],
                ["Change the CSV / demo / manual lab", "app/feature_page.py"],
                ["Change colors and the result banner look", "app/styles.py and app/widgets.py"],
                ["Change the receipt or the on-device warning", "src/privacy.py and app/privacy_ui.py"],
                ["Retrain", "src/train.py or src/train_audio.py, then replace the files in models/"],
            ],
            [4.15 * inch, 2.95 * inch],
        ),
        Spacer(1, 12),
        HRFlowable(width="100%", thickness=0.6, color=LINE, spaceAfter=8),
        P(
            "Landing page (landing/index.html) is only a public introduction. "
            "Vercel hosts that page. It does not load the models. The full app is the local Streamlit command at the start of this guide.",
            "Small",
        ),
        Spacer(1, 6),
        callout(
            "If a result worries you, or if speech, walking, or handwriting has changed in daily life, "
            "talk with a qualified clinician. Do not start or stop treatment from this screen.",
            GREEN,
        ),
    ]
    return flow


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.78 * inch,
        bottomMargin=0.72 * inch,
        title="VoiceSense — how the app works",
        author="VoiceSense",
    )
    doc.build(story(), onFirstPage=paint_first, onLaterPages=paint_later)
    print(OUT)


if __name__ == "__main__":
    main()
