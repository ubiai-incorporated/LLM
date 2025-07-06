# Startup Strategist – AI-Powered Business Ideation Assistant

Welcome to **Startup Strategist**, an AI-powered Streamlit chatbot that helps you brainstorm startup ideas, validate concepts using lean methodologies, generate lean canvases, save business documents, and send emails — all in a single app.

> 💡 Built with [LangChain](https://www.langchain.com/), [OpenAI](https://platform.openai.com/), and [Streamlit](https://streamlit.io/)

---

## Features

- 💡 **Brainstorm Startup Ideas** based on your interests, skills, and pain points.
- ✅ **Validate Ideas** using lean startup principles and structured checklists.
- 🧱 **Generate Lean Canvas** with all key components pre-filled.
- 📄 **Save to Word Document** for quick documentation and export.
- 📧 **Send Emails** to share results with collaborators or mentors.

---

## 📦 Requirements

Make sure you have **Python 3.9+** installed. Then install the dependencies:

```bash
pip install -r requirements.txt
```
✅ You’ll find the requirements.txt in this repository.

## 🔐 Configuration
### 1. OpenAI API Key
You must have an OpenAI API key to use GPT models. Update this line in the script with your key:

```python
llm = ChatOpenAI(model="gpt-4", openai_api_key="your_openai_api_key_here")
```
#### 2. Email Configuration (Optional)
If you'd like to enable email functionality, provide your email address and app password:



```python
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your_email@gmail.com",
    "sender_password": "your_app_password"
}
```

⚠️ Do NOT use your main email password. Generate an App Password instead if using Gmail.

▶️ How to Run

```bash
streamlit run streamlitapp.py
```
Replace app.py with the name of your Python file if different.

## 🧠 Using the Chatbot

Once the app launches:

1. Type your question or command in the chat input box (e.g., _"Brainstorm startup ideas related to education and AI"_).
2. The AI will respond in the chat area with actionable outputs.
3. Use the sidebar settings to configure your email credentials if needed.
4. The assistant will remember your conversation context and generate responses accordingly.
