# Hotel Enterprise Analytics

A comprehensive enterprise-grade forecasting and analytics platform for hotel management.

## Quick Start

1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd hotel_enterprise
   ```

2. **Set up the virtual environment** and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   # Open .env and add your HuggingFace/Anthropic API keys if needed
   ```

4. **Run the Backend (FastAPI)**:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

5. **Run the Frontend (Streamlit)**:
   ```bash
   streamlit run frontend/app.py
   ```

## Architecture Diagram

```mermaid
graph TD
    A[Frontend (Streamlit)] -->|HTTP GET/POST| B[Backend (FastAPI)]
    B -->|Model Inference| C[ML Models (Prophet, LightGBM)]
    B -->|NLP Tasks| D[HuggingFace / Anthropic API]
    C -->|Metrics & Tracking| E[MLflow]
    B -->|Cache / Data| F[(Data / Cache)]
```

- **Frontend**: A sleek, user-friendly Streamlit dashboard providing interactive visualizations.
- **Backend**: A modular FastAPI service managing routing, API integration, and model orchestration.
- **MLflow**: Tracks experiments, model parameters, and training metrics automatically.

## Project Structure

```
hotel_enterprise/
├── backend/                # FastAPI backend service
│   ├── main.py             # Entry point for backend
│   └── routers/            # API Endpoints (e.g., sentiment.py)
├── frontend/               # Streamlit frontend service
│   └── app.py              # Main dashboard application
├── src/                    # Core logic and ML engine
│   ├── sentiment_engine.py # NLP orchestration & fallback logic
│   ├── knowledge_distillation.py
│   └── train_models_ts.py  # Model training scripts
├── mlruns/                 # (Ignored) MLflow local tracking database
├── notebooks/              # Jupyter notebooks for experimentation
├── results/                # Visual evidence and MLflow dashboard plots
├── data/                   # Datasets and caches
├── requirements.txt        # Project dependencies
└── README.md               # You are here
```

## Features

- **Robust NLP Engine**: 3-tier fallback architecture (HuggingFace -> Anthropic Claude -> TextBlob) with comprehensive error handling.
- **Experiment Tracking**: Full MLflow integration recording parameters, metrics, and models.
- **Clean API Design**: Modularized FastAPI endpoints returning consistent responses.
