# legal-assistant-qdrant

## ⚖️ AI-Powered Legal Assistant

An interactive **legal research assistant** that leverages **Qdrant vector database** for semantic retrieval of Indian Supreme Court judgments and **Groq LLM** for context-aware summarization.  
Built to help lawyers, researchers, and students quickly summarize relevant legal cases using natural language queries.

** [Notebook](https://colab.research.google.com/drive/1sm16nqse-yE2XlajKRTw8Us597y536DG?usp=sharing) **
---

## 🌟 Features

- **Semantic Retrieval**: Uses LegalBERT embeddings to perform dense similarity search in Qdrant.  
- **Context-Aware Summarization**: Groq LLM generates human-readable summaries with citations.  
- **Filters**: Filter results by jurisdiction, year range, and case name.  
- **RAG Pipeline**: Retrieval-Augmented Generation with Qdrant as knowledge base.  
- **Streamlit Interface**: Simple web UI for querying and displaying results.  
- **Citations**: Auto-inserts `[n]` style references for retrieved cases.  

---

## 🗂 Dataset

- **Source**: [Kaggle: SC Judgments India 1950–2024](https://www.kaggle.com/datasets/adarshsingh0903/legal-dataset-sc-judgments-india-19502024/data)  
- **Preprocessing**:  
  - Extract text from PDFs using PyMuPDF  
  - Chunk into 512–1024 token sections  
  - Generate embeddings with `law-ai/InLegalBERT`  
  - Store embeddings and metadata in Qdrant  

---

## ⚙️ Installation

1. **Clone this repo**

```bash
git clone https://github.com/Jeet009/legal-assistant.git
cd legal-assistant

conda create -n legal-assistant python=3.10 -y
conda activate legal-assistant

pip install -r requirements.txt

export GROQ_API_KEY="your_groq_api_key"
export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY="your_qdrant_api_key"
```
## 📚 References

- [Qdrant Vector DB](https://qdrant.tech/)  
- [Groq LLM API](https://www.groq.ai/)  
- [LegalBERT](https://huggingface.co/law-ai/InLegalBERT)  
- [Kaggle Dataset: Indian Supreme Court Judgments](https://www.kaggle.com/datasets/adarshsingh0903/legal-dataset-sc-judgments-india-19502024/data)  

---

## ⚠️ Disclaimer

This tool is **for educational and research purposes only**. It **does not constitute legal advice**. Users should verify all AI-generated summaries against official legal sources.  

---

## 📄 License

MIT License © 2025 Jeet Mukherjee

