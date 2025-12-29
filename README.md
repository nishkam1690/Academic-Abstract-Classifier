## Academic Abstract Classifier

Build a lightweight GenAI-inspired service that predicts the field of study (AI, healthcare, business, environmental science, etc.) from an academic paper abstract. The prototype relies on the ArXiv Abstracts dataset (a small sample is bundled for convenience) and exposes predictions through a FastAPI-powered web app.

### Project Layout

- `data/` – sample ArXiv-style abstracts for quick experimentation.
- `models/` – serialized scikit-learn pipeline artifacts.
- `src/train_classifier.py` – training script that produces the artifacts.
- `app/main.py` – FastAPI application providing classification endpoints.

### Quickstart

1. **Install dependencies**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train / refresh the model**

   ```bash
   python src/train_classifier.py --data-path data/sample_arxiv.csv --model-dir models
   ```

3. **Run the web app / API**

   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

4. **Use the UI or endpoint**

   - Visit `http://127.0.0.1:8000/` for the HTML interface (paste an abstract, hit *Classify*).
   - Swagger docs live at `http://127.0.0.1:8000/docs`.

   To test the JSON endpoint directly:

   ```bash
   curl -X POST http://localhost:8000/classify \
        -H "Content-Type: application/json" \
        -d "{\"abstract\": \"We introduce a transformer-based approach for few-shot reasoning.\"}"
   ```

### Dataset Notes

- The real ArXiv Abstracts dataset can be sourced via Kaggle or ArXiv APIs.
- Replace `data/sample_arxiv.csv` with a richer dataset (thousands of rows) for meaningful accuracy.
- Update the label column `field` to cover the domain taxonomy you care about.

### Extending the Prototype

- Swap the TF-IDF + Logistic Regression baseline with transformer embeddings.
- Add confidence intervals, rationale snippets, or few-shot prompt-based backups.
- Deploy on managed services (Azure App Service, AWS App Runner, etc.) with CI/CD.


