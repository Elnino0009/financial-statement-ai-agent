# 🤖 Financial Statement AI Agent Platform

A sophisticated multi-agent AI system that analyzes financial documents (bank statements, investment factsheets) and provides intelligent insights through natural language conversations. Built with Python FastAPI backend and Next.js 15 frontend.

<img width="975" alt="Screenshot 2025-06-22 at 11 35 19 PM" src="https://github.com/user-attachments/assets/3e2237b6-4655-4611-98c6-fdfdc6667e8e" />

## 🎯 What This Project Does

This platform combines multiple AI agents to provide comprehensive financial analysis:

1. **Statement Analyst Agent** - Analyzes uploaded bank statements using RAG (Retrieval-Augmented Generation)
2. **Factsheet Analyst Agent** - Searches and analyzes investment factsheets using vector embeddings
3. **News Agent** - Fetches relevant financial news and market updates
4. **Orchestrator Agent** - Coordinates all agents to provide comprehensive responses

### Key Features

- 📄 **PDF Document Processing** - Upload and analyze bank statements and factsheets
- 💬 **Intelligent Chat Interface** - Ask questions about your financial documents in natural language
- 🔍 **Vector Search** - Advanced document search using FAISS and text embeddings
- 📊 **Multi-Agent Coordination** - Different specialized agents work together seamlessly
- 🌐 **Real-time Updates** - WebSocket support for live analysis progress
- 🎨 **Modern UI** - Beautiful, responsive interface built with Next.js 15 and Tailwind CSS

## 🧠 Key AI/ML Concepts Explained

If you're new to AI and want to understand the concepts used in this project, here are some helpful resources:

### Retrieval-Augmented Generation (RAG)
- **What it is**: [RAG Explained Simply](https://www.pinecone.io/learn/retrieval-augmented-generation/) - A technique that combines your documents with AI to answer questions accurately
- **Why it matters**: [The Complete Guide to RAG](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) - Helps AI provide factual answers based on your specific documents

### Vector Embeddings & Semantic Search
- **Beginner's Guide**: [Vector Embeddings Explained](https://www.pinecone.io/learn/vector-embeddings/) - How computers understand the meaning of text
- **Visual Explanation**: [Embeddings in Plain English](https://jalammar.github.io/illustrated-word2vec/) - See how words become numbers that capture meaning

### Multi-Agent AI Systems
- **Introduction**: [What are AI Agents?](https://www.anthropic.com/news/building-effective-agents) - Understanding how multiple AI agents work together
- **Architecture**: [Multi-Agent Systems Overview](https://towardsdatascience.com/multi-agent-systems-a-comprehensive-guide-4e8b8e5b5b1e) - How different agents coordinate to solve complex problems

### FAISS (Facebook AI Similarity Search)
- **Documentation**: [FAISS Overview](https://github.com/facebookresearch/faiss/wiki) - Fast similarity search for vector embeddings
- **Tutorial**: [FAISS for Beginners](https://www.pinecone.io/learn/faiss/) - How to search through millions of documents quickly

### LangChain Framework
- **Getting Started**: [LangChain Introduction](https://python.langchain.com/docs/get_started/introduction) - Framework for building AI applications
- **Concepts**: [LangChain Conceptual Guide](https://python.langchain.com/docs/concepts) - Understanding chains, agents, and tools

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   AI Services   │
│   (Next.js 15) │◄──►│   (FastAPI)      │◄──►│   (OpenAI API)  │
│                 │    │                  │    │                 │
│ • Chat UI       │    │ • Multi-Agent    │    │ • GPT-4.1 Mini  │
│ • File Upload   │    │   System         │    │ • Text Embedding│
│ • Progress      │    │ • WebSocket      │    │                 │
│   Tracking      │    │ • Document       │    │                 │
└─────────────────┘    │   Processing     │    └─────────────────┘
                       │                  │    
                       │ ┌──────────────┐ │    ┌─────────────────┐
                       │ │ Vector Store │ │◄──►│   External APIs │
                       │ │   (FAISS)    │ │    │                 │
                       │ └──────────────┘ │    │ • News API      │
                       └──────────────────┘    │ • Alpha Vantage │
                                              └─────────────────┘
```

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 18+** - [Download Node.js](https://nodejs.org/)
- **Git** - [Download Git](https://git-scm.com/downloads)

### Required API Keys

You'll need to obtain these free API keys:

1. **OpenAI API Key** - [Get it here](https://platform.openai.com/api-keys)
   - Used for AI chat completions and text embeddings
   - Required for all AI functionality

2. **News API Key** (Optional) - [Get it here](https://newsapi.org/)
   - Used for fetching financial news
   - Free tier: 1000 requests/month

3. **Alpha Vantage API Key** (Optional) - [Get it here](https://www.alphavantage.co/support/#api-key)
   - Used for stock market data
   - Free tier: 25 requests/day

## 📋 Installation Steps

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd financial-ai-platform
```

### Step 2: Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create environment variables file:**
   ```bash
   cp env.example .env  # Copy the example file and rename it
   ```

5. **Configure your `.env` file** with your API keys:
   ```env
   # Required
   OPENAI_API_KEY=your_openai_api_key_here

   # Optional - for news features
   NEWS_API_KEY=your_news_api_key_here
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
   
   # Optional - for advanced vector storage
   VECTORIZE_API_TOKEN=your_vectorize_api_token_here
   ```

6. **Start the backend server:**
   ```bash
   python main.py
   ```

   The backend will start at `http://localhost:8000`

### Step 3: Frontend Setup

1. **Open a new terminal** and navigate to frontend directory:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

   The frontend will start at `http://localhost:3000`

### Step 4: Verify Installation

1. Open your browser and go to `http://localhost:3000`
2. You should see the Financial AI Platform interface
3. Try uploading a sample PDF document to test the system

## 📱 How to Use

### Uploading Documents

1. **Bank Statements**: Click the "Upload Statement" button and select a PDF bank statement
2. **Investment Factsheets**: Use the "Upload Factsheet" section for investment documents

### Asking Questions

Once documents are uploaded, you can ask questions like:

- "What's my account balance?"
- "Show me my largest transactions this month"
- "What fees were charged to my account?"
- "Analyze my investment portfolio"
- "What's the latest news about Apple stock?"

### Understanding Responses

The AI will provide detailed answers and indicate which document or source the information came from.

## 🛠️ Technology Stack

**Backend:**
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [LangChain](https://python.langchain.com/) - LLM application framework
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [OpenAI API](https://openai.com/api/) - GPT-4.1 Mini and embeddings

**Frontend:**
- [Next.js 15](https://nextjs.org/) - React framework with App Router
- [React 19](https://react.dev/) - UI library
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
- [Shadcn/ui](https://ui.shadcn.com/) - Modern component library

**AI/ML:**
- **Text Embeddings**: `text-embedding-3-small` for document vectorization
- **Chat Completions**: `gpt-4.1-mini-2025-04-14` for intelligent responses
- **Vector Storage**: FAISS for fast similarity search

## 📂 Project Structure

```
financial-ai-platform/
├── backend/
│   ├── agents/                 # AI agent implementations
│   │   ├── orchestrator.py    # Coordinates all agents
│   │   ├── statement_analyst.py # Bank statement analysis
│   │   ├── factsheet_analyst.py # Investment document analysis
│   │   └── news_agent.py      # Financial news fetching
│   ├── storage/               # Document storage
│   │   ├── statements/        # Processed bank statements
│   │   └── factsheets/       # Processed factsheets
│   ├── main.py               # FastAPI server
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/              # Next.js app router pages
│   │   ├── components/       # React components
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── AnalysisResults.tsx
│   │   │   └── ui/           # Shadcn/ui components
│   │   └── hooks/            # React hooks
│   ├── package.json          # Node.js dependencies
│   └── next.config.ts        # Next.js configuration
└── README.md                 # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional - News Features
NEWS_API_KEY=your-news-api-key
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key

# Optional - Advanced Vector Storage
VECTORIZE_API_TOKEN=your-vectorize-token
VECTORIZE_ORG_ID=your-org-id
VECTORIZE_PIPELINE_ID=your-pipeline-id
```

### Customization Options

1. **Model Selection**: Change the AI model in `agents/orchestrator.py`
2. **Embedding Model**: Modify the embedding model in `agents/statement_analyst.py`
3. **UI Theme**: Customize colors in `frontend/src/app/globals.css`

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   - Ensure virtual environment is activated: `source venv/bin/activate`
   - Reinstall dependencies: `pip install -r requirements.txt`

2. **OpenAI API errors**:
   - Verify your API key is correct
   - Check your OpenAI account has sufficient credits
   - Ensure the key has proper permissions

3. **Frontend not connecting to backend**:
   - Verify backend is running on `http://localhost:8000`
   - Check CORS settings in `main.py`

4. **PDF processing fails**:
   - Ensure the PDF is not password-protected
   - Try with a smaller PDF file first
   - Check file permissions

### Getting Help

If you encounter issues:

1. Check the terminal/console for error messages
2. Verify all API keys are correctly set
3. Ensure all dependencies are installed
4. Try restarting both frontend and backend servers

## 🚀 Deployment

For production deployment:

1. **Backend**: Deploy to services like Railway, Render, or AWS
2. **Frontend**: Deploy to Vercel, Netlify, or similar platforms
3. **Environment Variables**: Set all required API keys in your deployment platform

## 🤝 Contributing

This is a learning project demonstrating multi-agent AI systems. Feel free to:

- Fork the repository
- Create feature branches
- Submit pull requests
- Report issues

## 📄 License

This project is intended for educational purposes. Please ensure you comply with the terms of service of all third-party APIs used.

## 🎓 Learning Resources

### Next Steps to Extend This Project

1. **Add More Agent Types**: Create agents for different financial tasks
2. **Implement Caching**: Add Redis for better performance
3. **Add Authentication**: Implement user accounts and security
4. **Mobile App**: Create a React Native version
5. **Advanced Analytics**: Add charts and data visualization

### Related Learning Materials

- [Building AI Agents with LangChain](https://python.langchain.com/docs/tutorials/agents)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Next.js 15 Documentation](https://nextjs.org/docs)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)

---

Built with ❤️ using AI, Python, and TypeScript. Perfect for learning about multi-agent systems and modern web development! 
