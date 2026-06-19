# AI Assistant Chat Application

A modern, feature-rich AI chat application built with Flask and powered by OpenRouter API. Supports text conversations, image analysis, PDF processing, and web search capabilities.

## 🌟 Features

- **💬 Natural Language Conversations**: Chat with various AI models through OpenRouter
- **🖼️ Image Analysis**: Upload and analyze images with vision models
- **📄 PDF Processing**: Extract text from PDFs and analyze documents
- **🔍 Web Search**: Search the web using `/search` commands
- **📱 Responsive Design**: Modern, mobile-friendly dark theme interface
- **💾 Persistent History**: SQLite database stores all conversations
- **🔒 Secure**: Environment-based configuration with proper error handling

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenRouter API key ([Get one here](https://openrouter.ai/))
- Optional: SerpAPI key for web search functionality

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd chat
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy the template and edit with your values
   cp env_template.txt .env
   ```
   
   Edit `.env` with your API keys:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   SECRET_KEY=your_secret_key_here
   SERPAPI_KEY=your_serpapi_key_here  # Optional
   ```

4. **Run the application**
   ```bash
   python chat_bot.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:9090`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key | - | ✅ Yes |
| `SECRET_KEY` | Flask secret key | Auto-generated | ❌ No |
| `FLASK_DEBUG` | Enable debug mode | `False` | ❌ No |
| `PORT` | Server port | `9090` | ❌ No |
| `DATABASE_PATH` | SQLite database path | `chat_history.db` | ❌ No |
| `UPLOAD_FOLDER` | File upload directory | `./uploads` | ❌ No |
| `MAX_CONTENT_LENGTH` | Max file size in bytes | `16777216` (16MB) | ❌ No |
| `SERPAPI_KEY` | SerpAPI key for web search | - | ❌ No |
| `DEFAULT_MODEL` | Default AI model | `deepseek/deepseek-chat-v3-0324:free` | ❌ No |

### Supported File Types

- **Images**: PNG, JPG, JPEG, GIF (up to 16MB)
- **Documents**: PDF (up to 16MB)

### Available AI Models

The application supports various models through OpenRouter:
- DeepSeek Chat v3 (default)
- Llama 4 Maverick
- Vision models for image analysis
- And many more available through OpenRouter

## 📖 Usage

### Basic Chat
1. Click "New Chat" to start a conversation
2. Type your message and press Enter or click Send
3. View AI responses in real-time

### Image Analysis
1. Click the "Attach File" button
2. Select an image file
3. Add a message or let the AI describe the image
4. Send to get AI analysis

### PDF Processing
1. Click "Attach File" and select a PDF
2. The AI will extract and analyze the text content
3. Ask questions about the document

### Web Search
Use the `/search` command followed by your query:
```
/search latest news about AI
```

### Session Management
- **Create**: Click "New Chat" to start fresh
- **Switch**: Click any conversation in the sidebar
- **Rename**: Click the edit icon next to session names
- **Delete**: Click the delete icon to remove conversations

## 🏗️ Architecture

### Backend (Flask)
- **`chat_bot.py`**: Main Flask application
- **Database**: SQLite with optimized queries and indexing
- **File Handling**: Secure upload processing with validation
- **API Integration**: OpenRouter for AI models, SerpAPI for search

### Frontend (HTML/CSS/JS)
- **Responsive Design**: Mobile-first approach
- **Dark Theme**: Modern, easy-on-eyes interface
- **Real-time Updates**: Dynamic message loading
- **File Preview**: Image thumbnails and PDF indicators

### Database Schema
```sql
-- Sessions table
sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP,
    last_updated TIMESTAMP,
    model TEXT NOT NULL
)

-- Messages table
messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP,
    has_image BOOLEAN,
    image_data TEXT,
    image_mime_type TEXT,
    has_pdf BOOLEAN,
    pdf_data TEXT,
    pdf_text TEXT
)
```

## 🛠️ Development

### Project Structure
```
chat/
├── chat_bot.py              # Main Flask application
├── llama_csv_analysis.py    # CSV analysis utility
├── templates/
│   └── index.html           # Frontend interface
├── requirements.txt         # Python dependencies
├── env_template.txt         # Environment variables template
├── README.md               # This file
└── chat_history.db         # SQLite database (created automatically)
```

### Running in Development
```bash
# Enable debug mode
export FLASK_DEBUG=True
python chat_bot.py
```

### Adding New Features
1. Backend changes go in `chat_bot.py`
2. Frontend changes go in `templates/index.html`
3. Update requirements.txt if adding new dependencies
4. Update this README for user-facing changes

## 🔒 Security Features

- ✅ Environment-based configuration (no hardcoded secrets)
- ✅ Input validation and sanitization
- ✅ File type and size restrictions
- ✅ SQL injection prevention with parameterized queries
- ✅ Secure filename handling
- ✅ Error handling without information leakage
- ✅ CSRF protection through Flask sessions

## 🚀 Deployment

### Production Considerations
1. **Use a production WSGI server** (not Flask's dev server):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:9090 chat_bot:app
   ```

2. **Set environment variables securely**
3. **Use HTTPS** in production
4. **Regular database backups**
5. **Monitor logs** for errors and usage

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 9090
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:9090", "chat_bot:app"]
```

## 🐛 Troubleshooting

### Common Issues

**"OPENROUTER_API_KEY environment variable is required"**
- Make sure you've set the API key in your `.env` file or environment

**File upload fails**
- Check file size (max 16MB by default)
- Verify file type is supported (images: PNG/JPG/JPEG/GIF, documents: PDF)

**Database errors**
- Ensure the application has write permissions in the directory
- Check disk space for SQLite database

**Vision models fail**
- Some vision models may be temporarily unavailable
- The app will try multiple models automatically

### Logs
Check the console output for detailed error messages and debugging information.

## 📄 License

This project is provided as-is for educational and development purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Open an issue with detailed information about the problem

---

**Happy chatting! 🤖✨**
