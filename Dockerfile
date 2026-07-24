FROM python:3.14-slim

# tesseract-ocr + deu: OCR in importDocuments_structural.py (pytesseract)
# default-jre-headless: language_tool_python.LanguageTool("de-CH") runs a local Java server
# libreoffice: soffice --headless, used by importDocuments_structural.py:pptx_to_pdf()
# poppler-utils: pdf2image (tests/test_read_faultypdf.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    default-jre-headless \
    libreoffice \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# torch==2.10.0 in requirements.txt is a macOS wheel and won't resolve here.
# Install everything else first, then torch separately for the target platform.
RUN grep -v '^torch==' requirements.txt > requirements-no-torch.txt \
    && pip install --no-cache-dir -r requirements-no-torch.txt

# TODO before deploying to the Spark: replace this with the correct CUDA/ARM64 build
# for DGX Spark's GB10 (Grace-Blackwell). Check https://pytorch.org/get-started/locally/
# (Linux / ARM64 / CUDA) or NVIDIA's NGC PyTorch container for a Blackwell-compatible
# build — the generic pip wheel below may not have Blackwell (sm_121) kernels yet.
RUN pip install --no-cache-dir torch

# Bake the LanguageTool download (~200MB) into the image so containers don't
# fetch it on first run.
RUN python -c "import language_tool_python; language_tool_python.LanguageTool('de-CH')"

COPY . .

# gui.py is the app's front door — it shells out to rag_answer_reference_facts.py,
# rag_append_case.py, importDocuments_structural.py, embed_e5.py, and
# llm_error_detector.py via subprocess, all of which live alongside it.
EXPOSE 8501
CMD ["streamlit", "run", "gui.py", "--server.port=8501", "--server.address=0.0.0.0"]
