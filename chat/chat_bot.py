from flask import Flask, render_template, request, jsonify, session, send_file
import os
import base64
import uuid
import sqlite3
import json
import logging
import time
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from openai import OpenAI
from functools import wraps
from pathlib import Path
from io import BytesIO
from serpapi import GoogleSearch  # For web search functionality
import fitz  # PyMuPDF for PDF processing
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    import secrets
    secret_key = secrets.token_urlsafe(32)
    logger.warning("SECRET_KEY not set. Using generated key. Set SECRET_KEY environment variable for production.")
app.secret_key = secret_key
app.permanent_session_lifetime = timedelta(days=31)

# Initialize OpenRouter client
api_key = os.environ.get('OPENROUTER_API_KEY')
if not api_key:
    logger.error("OPENROUTER_API_KEY environment variable is not set")
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# Configure upload settings
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', './uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # Default 16MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# Database setup
DATABASE_PATH = os.environ.get('DATABASE_PATH', 'chat_history.db')
DATABASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATABASE_PATH)

# Enhanced system prompt for better responses
SYSTEM_PROMPT = """You are a helpful, knowledgeable AI assistant. You are:

🎯 Core Traits:
- Helpful, friendly, respectful
- Thorough yet concise
- Patient and understanding

💡 Response Guidelines:
- Provide accurate, well-researched information
- Use clear formatting with short headings and bullet points when helpful
- Maintain context and ask clarifying questions when needed
- Acknowledge uncertainty when applicable

🔄 Conversation Flow:
- Remember previous messages and build upon them
- Reference earlier parts of the conversation when relevant
- Offer follow-up suggestions or related topics when appropriate

Always strive to be genuinely helpful while keeping interactions efficient and productive."""

# Default models with fallbacks
DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'openai/gpt-4o-mini')

# Text chat models with fallbacks (order matters - first is preferred)
TEXT_CHAT_MODELS = [
    DEFAULT_MODEL
]

# Vision models with fallbacks (order matters - first is preferred)
VISION_MODELS = [
    DEFAULT_MODEL
]

# MIME type mapping
MIME_TYPES = {
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'gif': 'image/gif',
    'pdf': 'application/pdf'
}

# Response enhancement settings
RESPONSE_SETTINGS = {
    'max_tokens': 4096,
    'temperature': 0.7,
    'top_p': 0.9,
    'frequency_penalty': 0.1,
    'presence_penalty': 0.1
}

def get_db_connection():
    """Create a database connection with row factory and optimizations."""
    try:
        conn = sqlite3.connect(DATABASE, timeout=20.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable foreign key constraints
        conn.execute('PRAGMA foreign_keys = ON')
        # Performance optimizations
        conn.execute('PRAGMA journal_mode = WAL')  # Write-Ahead Logging for better concurrency
        conn.execute('PRAGMA synchronous = NORMAL')  # Balance between safety and speed
        conn.execute('PRAGMA cache_size = -64000')  # 64MB cache
        conn.execute('PRAGMA temp_store = MEMORY')  # Store temporary tables in memory
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def check_column_exists(conn, table, column):
    """Check if a column exists in a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns

def extract_text_from_pdf(pdf_data):
    """Extract text content from PDF file data.
    
    Args:
        pdf_data (bytes): PDF file data
    Returns:
        str: Extracted text content
    """
    if not pdf_data:
        return "Error: No PDF data provided"
        
    try:
        # Open PDF from memory
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        
        if not pdf_document:
            return "Error: Could not open PDF document"
            
        if pdf_document.page_count == 0:
            pdf_document.close()
            return "Error: PDF document has no pages"
        
        text_content = ""
        
        # Extract text from each page
        for page_num in range(pdf_document.page_count):
            try:
                page = pdf_document[page_num]
                if page:
                    page_text = page.get_text()
                    if page_text:
                        text_content += f"Page {page_num + 1}:\n{page_text}\n\n"
            except Exception as page_error:
                logger.warning(f"Error extracting text from page {page_num + 1}: {str(page_error)}")
                text_content += f"Page {page_num + 1}: [Error extracting text]\n\n"
        
        pdf_document.close()
        
        final_text = text_content.strip()
        if not final_text:
            return "Warning: PDF appears to be empty or contains no extractable text (might be image-based PDF)"
            
        return final_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return f"Error reading PDF: {str(e)}"

def migrate_database():
    """Migrate database schema to latest version."""
    with get_db_connection() as conn:
        # Check if image_mime_type column exists in messages table
        if not check_column_exists(conn, 'messages', 'image_mime_type'):
            logger.info("Adding image_mime_type column to messages table")
            conn.execute('ALTER TABLE messages ADD COLUMN image_mime_type TEXT')
            conn.commit()
        
        # Check if other columns exist and add them if needed
        if not check_column_exists(conn, 'messages', 'has_image'):
            logger.info("Adding has_image column to messages table")
            conn.execute('ALTER TABLE messages ADD COLUMN has_image BOOLEAN DEFAULT 0')
            conn.commit()
        
        if not check_column_exists(conn, 'messages', 'image_data'):
            logger.info("Adding image_data column to messages table")
            conn.execute('ALTER TABLE messages ADD COLUMN image_data TEXT')
            conn.commit()
        
        # Check if PDF-related columns exist and add them if needed
        if not check_column_exists(conn, 'messages', 'has_pdf'):
            logger.info("Adding has_pdf column to messages table")
            conn.execute('ALTER TABLE messages ADD COLUMN has_pdf BOOLEAN DEFAULT 0')
            conn.commit()
        
        if not check_column_exists(conn, 'messages', 'pdf_data'):
            logger.info("Adding pdf_data column to messages table")
            conn.execute('ALTER TABLE messages ADD COLUMN pdf_data TEXT')
            conn.commit()
        
        if not check_column_exists(conn, 'messages', 'pdf_text'):
            logger.info("Adding pdf_text column to messages table")
            conn.execute('ALTER TABLE messages ADD COLUMN pdf_text TEXT')
            conn.commit()

def init_db():
    """Initialize the database tables if they don't exist."""
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    
    with get_db_connection() as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model TEXT NOT NULL
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            has_image BOOLEAN DEFAULT 0,
            image_data TEXT,
            image_mime_type TEXT,
            has_pdf BOOLEAN DEFAULT 0,
            pdf_data TEXT,
            pdf_text TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        )
        ''')
        
        # Create indexes for better query performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_sessions_last_updated ON sessions (last_updated DESC)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages (timestamp)')
        conn.commit()
        logger.info("Database initialized successfully")
    
    # Run migrations after initial setup
    migrate_database()

def cleanup_empty_chats():
    """Delete sessions with no messages."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.id 
                FROM sessions s 
                LEFT JOIN messages m ON s.id = m.session_id 
                WHERE m.session_id IS NULL
            """)
            session_ids = [row['id'] for row in cursor.fetchall()]
            for session_id in session_ids:
                cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            deleted_count = len(session_ids)
            logger.info(f"Deleted {deleted_count} empty chats")
            return deleted_count
    except Exception as e:
        logger.error(f"Error deleting empty chats: {str(e)}")
        raise

def cleanup_old_sessions(days_old=30):
    """Delete sessions older than specified days with few messages."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now() - timedelta(days=days_old)).strftime('%Y-%m-%d')
            
            # Delete sessions older than cutoff with less than 3 messages
            cursor.execute("""
                DELETE FROM sessions 
                WHERE id IN (
                    SELECT s.id 
                    FROM sessions s 
                    LEFT JOIN (
                        SELECT session_id, COUNT(*) as msg_count 
                        FROM messages 
                        GROUP BY session_id
                    ) m ON s.id = m.session_id 
                    WHERE s.last_updated < ? AND (m.msg_count IS NULL OR m.msg_count < 3)
                )
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Deleted {deleted_count} old inactive chats")
            return deleted_count
    except Exception as e:
        logger.error(f"Error deleting old chats: {str(e)}")
        return 0

def handle_model_error(error_msg, model_name):
    """Handle and categorize model errors for better user feedback."""
    error_lower = error_msg.lower()
    
    if any(term in error_lower for term in ['rate limit', 'too many requests', '429']):
        return {
            'type': 'rate_limit',
            'message': f"⏱️ **Rate Limit Reached**\n\nThe {model_name} model is currently experiencing high demand. Please wait a moment and try again.",
            'retry_after': 30
        }
    elif any(term in error_lower for term in ['503', 'no instances', 'unavailable', 'service']):
        return {
            'type': 'service_unavailable', 
            'message': f"🔧 **Service Temporarily Unavailable**\n\nThe {model_name} model is temporarily down for maintenance or experiencing high load.",
            'retry_after': 60
        }
    elif any(term in error_lower for term in ['timeout', 'time out', 'deadline']):
        return {
            'type': 'timeout',
            'message': f"⏰ **Request Timeout**\n\nThe {model_name} model took too long to respond. This usually resolves quickly.",
            'retry_after': 15
        }
    elif any(term in error_lower for term in ['token', 'context', 'length', 'too long']):
        return {
            'type': 'context_length',
            'message': f"📏 **Message Too Long**\n\nYour message or conversation history is too long for the {model_name} model. Try shortening your message or starting a new conversation.",
            'retry_after': 0
        }
    else:
        return {
            'type': 'unknown',
            'message': f"❌ **Unexpected Error**\n\nThe {model_name} model encountered an unexpected issue. Our team has been notified.",
            'retry_after': 30
        }

# Initialize database on startup
init_db()

@app.route('/delete_empty_chats', methods=['POST'])
def delete_empty_chats():
    """Endpoint to manually delete empty chats."""
    try:
        deleted_count = cleanup_empty_chats()
        return jsonify({"success": True, "deleted_count": deleted_count})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/cleanup_old_sessions', methods=['POST'])
def cleanup_old_sessions_endpoint():
    """Endpoint to manually clean up old sessions."""
    try:
        days = request.json.get('days', 30) if request.json else 30
        deleted_count = cleanup_old_sessions(days)
        return jsonify({"success": True, "deleted_count": deleted_count})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/auto_rename_session/<session_id>', methods=['POST'])
def manual_auto_rename_session(session_id):
    """Endpoint to manually trigger auto-rename for a session."""
    try:
        if not validate_session_id(session_id):
            return jsonify({"error": "Invalid session ID"}), 400
        
        with get_db_connection() as conn:
            # Get all messages for the session
            messages_db = conn.execute(
                'SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC',
                (session_id,)
            ).fetchall()
            
            if not messages_db:
                return jsonify({"error": "No messages found for this session"}), 404
            
            # Generate new title
            new_title = analyze_conversation_for_title(messages_db, session_id)
            
            if new_title:
                conn.execute(
                    'UPDATE sessions SET title = ? WHERE id = ?',
                    (new_title, session_id)
                )
                conn.commit()
                
                # Get updated session data
                updated_session = conn.execute(
                    'SELECT * FROM sessions WHERE id = ?', 
                    (session_id,)
                ).fetchone()
                
                return jsonify({
                    "success": True, 
                    "new_title": new_title,
                    "session": dict(updated_session)
                })
            else:
                return jsonify({"error": "Could not generate a meaningful title"}), 400
                
    except Exception as e:
        logger.error(f"Error in manual auto-rename: {str(e)}")
        return jsonify({"error": f"Failed to rename session: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Test database connection
        with get_db_connection() as conn:
            conn.execute('SELECT 1').fetchone()
            # Get some stats
            session_count = conn.execute('SELECT COUNT(*) as count FROM sessions').fetchone()['count']
            message_count = conn.execute('SELECT COUNT(*) as count FROM messages').fetchone()['count']
        
        # Test OpenRouter API (simplified)
        api_status = "connected" if api_key else "no_api_key"
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "api": api_status,
            "default_model": DEFAULT_MODEL,
            "stats": {
                "total_sessions": session_count,
                "total_messages": message_count
            },
            "features": {
                "auto_rename": True,
                "context_management": True,
                "file_upload": True,
                "web_search": bool(os.environ.get('SERPAPI_KEY'))
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

def allowed_file(filename):
    """Check if the file extension is allowed."""
    if not filename or not isinstance(filename, str):
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_session_id(session_id):
    """Validate session ID format (UUID)."""
    if not session_id or not isinstance(session_id, str):
        return False
    try:
        uuid.UUID(session_id)
        return True
    except ValueError:
        return False

def sanitize_filename(filename):
    """Sanitize filename for security."""
    if not filename:
        return ""
    # Remove path separators and dangerous characters
    filename = os.path.basename(filename)
    return secure_filename(filename)

def get_mime_type(filename):
    """Get MIME type from filename."""
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return MIME_TYPES.get(extension, 'application/octet-stream')

def requires_valid_session(f):
    """Decorator to ensure a valid session exists."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session.permanent = True
        
        if 'current_session_id' not in session:
            new_session_id = str(uuid.uuid4())
            session['current_session_id'] = new_session_id
            
            with get_db_connection() as conn:
                conn.execute(
                    'INSERT INTO sessions (id, title, model) VALUES (?, ?, ?)',
                    (new_session_id, f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}", DEFAULT_MODEL)
                )
                conn.commit()
                logger.info(f"New session created: {new_session_id}")
        
        return f(*args, **kwargs)
    return decorated_function

def perform_search(query):
    """Perform a web search using SERPAPI.
    
    Args:
        query (str): The search query.
    Returns:
        dict: The search results.
    """
    api_key = os.environ.get('SERPAPI_KEY')
    if not api_key:
        raise ValueError("SERPAPI_KEY is not set in environment variables.")
    search = GoogleSearch({
        "q": query,
        "api_key": api_key
    })
    results = search.get_dict()
    return results

def format_search_results(results):
    """Format the search results for display in the chat.
    
    Args:
        results (dict): The search results from SERPAPI.
    Returns:
        str: Formatted search results.
    """
    organic_results = results.get('organic_results', [])
    if not organic_results:
        return "No search results found."
    formatted = "Here are the top search results:\n\n"
    for result in organic_results[:3]:
        title = result.get('title', 'No title')
        snippet = result.get('snippet', 'No snippet')
        formatted += f"**{title}**\n{snippet}\n\n"
    return formatted

def generate_title_from_message(message):
    """Generate a meaningful session title based on the first user message.
    
    Args:
        message (str): The first user message.
    Returns:
        str: A generated title.
    """
    # Clean and process the message
    message = message.strip()
    
    # Remove common chat prefixes
    prefixes_to_remove = ["/search", "please", "can you", "could you", "help me", "i need"]
    message_lower = message.lower()
    for prefix in prefixes_to_remove:
        if message_lower.startswith(prefix):
            message = message[len(prefix):].strip()
            break
    
    # Take meaningful words
    words = [w for w in message.split()[:6] if len(w) > 2]  # Skip very short words
    if not words:
        words = message.split()[:3]  # Fallback to first 3 words
    
    # Create title
    title = " ".join(words)
    
    # Add emoji based on content type
    message_lower = message.lower()
    if any(word in message_lower for word in ["code", "programming", "function", "debug"]):
        title = f"💻 {title}"
    elif any(word in message_lower for word in ["help", "question", "how", "what", "why"]):
        title = f"❓ {title}"
    elif any(word in message_lower for word in ["analysis", "analyze", "research", "study"]):
        title = f"📊 {title}"
    elif any(word in message_lower for word in ["creative", "story", "write", "poem"]):
        title = f"✍️ {title}"
    else:
        title = f"💬 {title}"
    
    return title[:50]  # Limit to 50 characters

def analyze_conversation_for_title(messages_db, session_id):
    """Analyze conversation content to generate an intelligent title.
    
    Args:
        messages_db: List of message records from database
        session_id: Session ID for logging
    
    Returns:
        str: Generated title or None if analysis fails
    """
    try:
        if not messages_db or len(messages_db) < 2:
            return None
        
        # Extract user messages and topics
        user_messages = [msg['content'] for msg in messages_db if msg['role'] == 'user']
        assistant_messages = [msg['content'] for msg in messages_db if msg['role'] == 'assistant']
        
        if not user_messages:
            return None
        
        # Analyze conversation patterns and topics
        conversation_text = " ".join(user_messages[:5])  # First 5 user messages
        conversation_lower = conversation_text.lower()
        
        # Define topic categories with keywords and emojis
        topic_categories = {
            'programming': {
                'keywords': ['code', 'python', 'javascript', 'programming', 'function', 'variable', 'debug', 'error', 'api', 'database', 'sql', 'html', 'css', 'react', 'nodejs', 'git', 'github'],
                'emoji': '💻',
                'titles': ['Coding Help', 'Programming Discussion', 'Development Support', 'Code Review', 'Tech Problem Solving']
            },
            'data_analysis': {
                'keywords': ['data', 'analysis', 'chart', 'graph', 'statistics', 'excel', 'csv', 'pandas', 'visualization', 'dataset', 'metrics', 'analytics'],
                'emoji': '📊',
                'titles': ['Data Analysis', 'Statistics Discussion', 'Data Visualization', 'Analytics Help', 'Research Analysis']
            },
            'writing': {
                'keywords': ['write', 'writing', 'story', 'poem', 'essay', 'article', 'content', 'creative', 'blog', 'script', 'letter', 'email'],
                'emoji': '✍️',
                'titles': ['Creative Writing', 'Writing Help', 'Content Creation', 'Story Development', 'Writing Support']
            },
            'learning': {
                'keywords': ['learn', 'study', 'understand', 'explain', 'tutorial', 'lesson', 'course', 'education', 'teaching', 'knowledge'],
                'emoji': '📚',
                'titles': ['Learning Session', 'Study Help', 'Educational Discussion', 'Knowledge Sharing', 'Tutorial Request']
            },
            'business': {
                'keywords': ['business', 'marketing', 'strategy', 'finance', 'investment', 'startup', 'company', 'management', 'sales', 'revenue'],
                'emoji': '💼',
                'titles': ['Business Discussion', 'Strategy Planning', 'Business Advice', 'Marketing Help', 'Business Analysis']
            },
            'research': {
                'keywords': ['research', 'study', 'investigation', 'survey', 'academic', 'paper', 'thesis', 'literature', 'sources', 'references'],
                'emoji': '🔍',
                'titles': ['Research Help', 'Academic Discussion', 'Research Planning', 'Literature Review', 'Study Research']
            },
            'health': {
                'keywords': ['health', 'medical', 'fitness', 'exercise', 'nutrition', 'diet', 'wellness', 'mental health', 'therapy'],
                'emoji': '🏥',
                'titles': ['Health Discussion', 'Wellness Advice', 'Fitness Help', 'Health Information', 'Medical Query']
            },
            'travel': {
                'keywords': ['travel', 'trip', 'vacation', 'destination', 'flight', 'hotel', 'tourism', 'adventure', 'journey'],
                'emoji': '✈️',
                'titles': ['Travel Planning', 'Trip Discussion', 'Travel Advice', 'Destination Info', 'Travel Help']
            }
        }
        
        # Find the most relevant topic
        max_score = 0
        best_category = None
        
        for category, data in topic_categories.items():
            score = sum(1 for keyword in data['keywords'] if keyword in conversation_lower)
            if score > max_score:
                max_score = score
                best_category = category
        
        # Generate title based on identified topic
        if best_category and max_score >= 2:  # At least 2 matching keywords
            category_data = topic_categories[best_category]
            
            # Try to extract specific topic from the first user message
            first_message = user_messages[0][:100]  # First 100 chars
            
            # Clean the message for title extraction
            clean_message = first_message.strip()
            for prefix in ['please', 'can you', 'could you', 'help me', 'i need', 'how to', 'what is']:
                if clean_message.lower().startswith(prefix):
                    clean_message = clean_message[len(prefix):].strip()
                    break
            
            # Extract key words from the clean message
            key_words = [word for word in clean_message.split()[:4] if len(word) > 2]
            
            if key_words:
                title = f"{category_data['emoji']} {' '.join(key_words).title()}"
            else:
                # Use a generic title from the category
                import random
                title = f"{category_data['emoji']} {random.choice(category_data['titles'])}"
        else:
            # Generate a general title from the conversation
            # Extract the most important words
            first_message = user_messages[0]
            
            # Remove common stop words but keep important ones
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            
            words = [word for word in first_message.split() if word.lower() not in stop_words and len(word) > 2]
            
            if words:
                title_words = words[:4]  # Take first 4 meaningful words
                title = f"💬 {' '.join(title_words).title()}"
            else:
                # Last resort: use message length to determine type
                if len(first_message) > 100:
                    title = "📝 Detailed Discussion"
                elif any(char in first_message for char in '?'):
                    title = "❓ Question & Answer"
                else:
                    title = "💬 Chat Session"
        
        # Ensure title isn't too long
        final_title = title[:50] if len(title) > 50 else title
        
        logger.info(f"Generated intelligent title for session {session_id}: '{final_title}'")
        return final_title
        
    except Exception as e:
        logger.error(f"Error analyzing conversation for title: {str(e)}")
        return None

def should_auto_rename_session(messages_db):
    """Determine if a session should be automatically renamed.
    
    Args:
        messages_db: List of message records
    
    Returns:
        bool: True if session should be renamed
    """
    if not messages_db:
        return False
    
    message_count = len([msg for msg in messages_db if msg['role'] in ['user', 'assistant']])
    
    # Auto-rename after 3 messages (user, assistant, user) or 5 total messages
    return message_count in [3, 5, 10]  # Rename at these message milestones

def auto_rename_session(session_id, messages_db):
    """Automatically rename a session based on conversation content.
    
    Args:
        session_id: ID of the session to rename
        messages_db: List of message records
    
    Returns:
        bool: True if session was renamed successfully
    """
    try:
        if not should_auto_rename_session(messages_db):
            return False
        
        new_title = analyze_conversation_for_title(messages_db, session_id)
        
        if new_title:
            with get_db_connection() as conn:
                # Check if current title is generic (auto-generated)
                current_session = conn.execute(
                    'SELECT title FROM sessions WHERE id = ?', 
                    (session_id,)
                ).fetchone()
                
                if current_session:
                    current_title = current_session['title']
                    
                    # Only auto-rename if current title looks generic
                    generic_patterns = [
                        'Chat 20',  # Date-based titles
                        '💬 ',  # Basic chat emoji titles
                        'Chat Session',
                        'New Chat'
                    ]
                    
                    is_generic = any(pattern in current_title for pattern in generic_patterns)
                    
                    if is_generic or len(current_title) < 15:  # Short titles are likely generic
                        conn.execute(
                            'UPDATE sessions SET title = ? WHERE id = ?',
                            (new_title, session_id)
                        )
                        conn.commit()
                        logger.info(f"Auto-renamed session {session_id} from '{current_title}' to '{new_title}'")
                        return True
                    else:
                        logger.info(f"Session {session_id} has custom title '{current_title}', skipping auto-rename")
        
        return False
        
    except Exception as e:
        logger.error(f"Error auto-renaming session {session_id}: {str(e)}")
        return False

def try_models_with_fallback(formatted_messages, models_list, max_retries=2, model_type="text"):
    """Try models with fallback and retry logic for both text and vision models.
    
    Args:
        formatted_messages: The formatted messages for the API
        models_list: List of models to try in order
        max_retries: Maximum number of retries per model
        model_type: Type of model ("text" or "vision")
    
    Returns:
        tuple: (response_content, used_model) or (None, None) if all fail
    """
    for model in models_list:
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting {model_type} model: {model} (attempt {attempt + 1})")
                
                # Enhanced parameters for better responses
                api_params = {
                    "extra_headers": {
                        "HTTP-Referer": request.headers.get('Origin', request.host_url),
                        "X-Title": f"Python Web Chatbot - {model_type.title()} Processing",
                    },
                    "model": model,
                    "messages": formatted_messages,
                    "timeout": 45,
                    "temperature": RESPONSE_SETTINGS.get('temperature', 0.7),
                    "max_tokens": RESPONSE_SETTINGS.get('max_tokens', 2048),
                }

                if model_type == "text":
                    api_params["top_p"] = RESPONSE_SETTINGS.get('top_p', 0.9)
                    api_params["frequency_penalty"] = RESPONSE_SETTINGS.get('frequency_penalty', 0.0)
                    api_params["presence_penalty"] = RESPONSE_SETTINGS.get('presence_penalty', 0.0)

                completion = client.chat.completions.create(**api_params)

                # Enhanced null checks and response validation
                if completion and completion.choices and len(completion.choices) > 0:
                    choice = completion.choices[0]
                    if choice and choice.message and choice.message.content:
                        response = choice.message.content.strip()
                        if response and len(response) > 5:  # Ensure meaningful response
                            logger.info(f"Successfully used {model_type} model: {model}")
                        return response, model
                    else:
                            logger.warning(f"Response too short from {model_type} model: {model}")
                            break  # Try next model
                else:
                        logger.warning(f"Empty response from {model_type} model: {model}")
                        break  # Try next model
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"{model_type.title()} model {model} failed (attempt {attempt + 1}): {error_msg}")
                
                if any(err in error_msg.lower() for err in ["503", "no instances available", "rate limit", "timeout"]):
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    continue
                
                # If it's a different error, try next model immediately
                break
    
    return None, None

# Keep the original function for backward compatibility
def try_vision_models(formatted_messages, max_retries=3):
    """Try vision models with fallback and retry logic."""
    return try_models_with_fallback(formatted_messages, VISION_MODELS, max_retries, "vision")

@app.route('/')
@requires_valid_session
def index():
    """Render the main chat page."""
    with get_db_connection() as conn:
        sessions = conn.execute('SELECT * FROM sessions ORDER BY last_updated DESC').fetchall()
    
    return render_template('index.html', 
                         sessions=sessions, 
                         current_session_id=session.get('current_session_id'))

@app.route('/get_session/<session_id>')
def get_session(session_id):
    """Retrieve session details and messages."""
    with get_db_connection() as conn:
        session_data = conn.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone()
        if not session_data:
            return jsonify({"error": "Session not found"}), 404
        
        # Use safer query to handle missing columns
        try:
            messages = conn.execute(
                'SELECT id, role, content, timestamp, has_image, image_data, image_mime_type, has_pdf, pdf_data, pdf_text FROM messages WHERE session_id = ? ORDER BY timestamp ASC', 
                (session_id,)
            ).fetchall()
        except sqlite3.OperationalError:
            # Fallback for older database schema
            try:
                messages = conn.execute(
                    'SELECT id, role, content, timestamp, has_image, image_data, image_mime_type, 0 as has_pdf, NULL as pdf_data, NULL as pdf_text FROM messages WHERE session_id = ? ORDER BY timestamp ASC', 
                    (session_id,)
                ).fetchall()
            except sqlite3.OperationalError:
                # Fallback for even older database schema
                messages = conn.execute(
                    'SELECT id, role, content, timestamp, 0 as has_image, NULL as image_data, NULL as image_mime_type, 0 as has_pdf, NULL as pdf_data, NULL as pdf_text FROM messages WHERE session_id = ? ORDER BY timestamp ASC', 
                    (session_id,)
                ).fetchall()
    
    session['current_session_id'] = session_id
    
    message_list = [{
        "id": msg['id'],
        "role": msg['role'],
        "content": msg['content'],
        "timestamp": msg['timestamp'],
        "has_image": bool(msg['has_image']),
        "image_data": msg['image_data'] if msg['has_image'] else None,
        "has_pdf": bool(msg['has_pdf'] if 'has_pdf' in msg.keys() else 0),
        "pdf_data": msg['pdf_data'] if ('has_pdf' in msg.keys() and msg['has_pdf']) else None
    } for msg in messages]
    
    return jsonify({
        "session": dict(session_data),
        "messages": message_list
    })

@app.route('/create_session', methods=['POST'])
@requires_valid_session
def create_session():
    """Create a new chat session."""
    data = request.json
    title = data.get('title', f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    model = data.get('model', DEFAULT_MODEL)
    new_session_id = str(uuid.uuid4())
    
    with get_db_connection() as conn:
        conn.execute(
            'INSERT INTO sessions (id, title, model) VALUES (?, ?, ?)',
            (new_session_id, title, model)
        )
        conn.commit()
        logger.info(f"New session created: {new_session_id}")
        new_session = conn.execute('SELECT * FROM sessions WHERE id = ?', (new_session_id,)).fetchone()
    
    session['current_session_id'] = new_session_id
    return jsonify({"session": dict(new_session), "messages": []})

@app.route('/update_session/<session_id>', methods=['POST'])
def update_session(session_id):
    """Update session properties like title or model."""
    try:
        # Validate session ID
        if not validate_session_id(session_id):
            return jsonify({"error": "Invalid session ID"}), 400
            
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate and sanitize updates
        updates = {}
        for k, v in data.items():
            if k in ['title', 'model'] and v:
                if k == 'title':
                    # Validate title
                    title = str(v).strip()
                    if not title:
                        return jsonify({"error": "Title cannot be empty"}), 400
                    if len(title) > 100:
                        return jsonify({"error": "Title too long (max 100 characters)"}), 400
                    updates[k] = title
                elif k == 'model':
                    # Validate model
                    model = str(v).strip()
                    if model:
                        updates[k] = model
        
        if not updates:
            return jsonify({"error": "No valid fields to update"}), 400
        
        updates['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with get_db_connection() as conn:
            # Check if session exists first
            existing_session = conn.execute('SELECT id FROM sessions WHERE id = ?', (session_id,)).fetchone()
            if not existing_session:
                return jsonify({"error": "Session not found"}), 404
            
            # Update the session
            set_clause = ', '.join(f"{k} = ?" for k in updates)
            params = list(updates.values()) + [session_id]
            
            result = conn.execute(
                f"UPDATE sessions SET {set_clause} WHERE id = ?",
                params
            )
            conn.commit()
            
            if result.rowcount == 0:
                return jsonify({"error": "Failed to update session"}), 500
            
            # Fetch updated session
            updated_session = conn.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone()
        
        logger.info(f"Session {session_id} updated successfully: {updates}")
        return jsonify({"session": dict(updated_session), "success": True})
        
    except Exception as e:
        logger.error(f"Error updating session {session_id}: {str(e)}")
        return jsonify({"error": f"Failed to update session: {str(e)}"}), 500

@app.route('/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session and its associated messages."""
    try:
        with get_db_connection() as conn:
            conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
            conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
            conn.commit()
            logger.info(f"Session deleted: {session_id}")
        
        if session.get('current_session_id') == session_id:
            session.pop('current_session_id', None)
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        return jsonify({"error": f"Failed to delete session: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
@requires_valid_session
def chat():
    """Handle text-based chat messages, including search commands."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', session.get('current_session_id'))
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
            
        if len(user_message) > 10000:  # Reasonable message length limit
            return jsonify({"error": "Message too long"}), 400
            
        if session_id and not validate_session_id(session_id):
            return jsonify({"error": "Invalid session ID"}), 400
    except (TypeError, ValueError) as e:
        return jsonify({"error": "Invalid request data"}), 400
    
    try:
        with get_db_connection() as conn:
            session_data = conn.execute('SELECT model FROM sessions WHERE id = ?', (session_id,)).fetchone()
            
            if not session_data:
                conn.execute(
                    'INSERT INTO sessions (id, title, model) VALUES (?, ?, ?)',
                    (session_id, f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                     data.get('model', DEFAULT_MODEL))
                )
                conn.commit()
                logger.info(f"New session created in chat: {session_id}")
                selected_model = data.get('model', DEFAULT_MODEL)
            else:
                selected_model = session_data['model']
            
            # Use safer query to handle missing columns
            try:
                messages_db = conn.execute(
                    'SELECT role, content, has_image, image_data, image_mime_type, has_pdf, pdf_data, pdf_text FROM messages WHERE session_id = ? ORDER BY timestamp ASC', 
                    (session_id,)
                ).fetchall()
            except sqlite3.OperationalError:
                # Fallback for older database schema
                try:
                    messages_db = conn.execute(
                        'SELECT role, content, has_image, image_data, image_mime_type, 0 as has_pdf, NULL as pdf_data, NULL as pdf_text FROM messages WHERE session_id = ? ORDER BY timestamp ASC', 
                        (session_id,)
                    ).fetchall()
                except sqlite3.OperationalError:
                    # Fallback for even older database schema
                    messages_db = conn.execute(
                        'SELECT role, content, 0 as has_image, NULL as image_data, NULL as image_mime_type, 0 as has_pdf, NULL as pdf_data, NULL as pdf_text FROM messages WHERE session_id = ? ORDER BY timestamp ASC', 
                        (session_id,)
                    ).fetchall()
            
            is_first_message = len(messages_db) == 0
            conn.execute(
                'INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)',
                (session_id, "user", user_message)
            )
            if is_first_message:
                new_title = generate_title_from_message(user_message)
                conn.execute(
                    'UPDATE sessions SET title = ? WHERE id = ?',
                    (new_title, session_id)
                )
            conn.commit()
            logger.info(f"User message added to session {session_id}")
        
            if user_message.startswith("/search"):
                query = user_message[len("/search"):].strip()
                if not query:
                    return jsonify({"error": "Please provide a search query after /search"}), 400
                search_results = perform_search(query)
                ai_response = f"Search results for '{query}':\n\n{format_search_results(search_results)}"
            else:
                formatted_messages = format_messages_for_api(messages_db, user_message)
                
                # Try text models with fallback
                models_to_try = [selected_model] + [m for m in TEXT_CHAT_MODELS if m != selected_model]
                logger.info(f"Processing chat with primary model: {selected_model}")
                
                ai_response, used_model = try_models_with_fallback(
                    formatted_messages, 
                    models_to_try,
                    max_retries=2,
                    model_type="text"
                )
                
                if used_model and used_model != selected_model:
                    logger.info(f"Fallback successful: used {used_model} instead of {selected_model}")
                
                # If all models failed, provide a helpful fallback with better error context
                if ai_response is None:
                    ai_response = """I apologize, but I'm experiencing technical difficulties right now. 🤖

**Current Status:**
• Primary model ({}) is temporarily unavailable
• All backup models are also experiencing issues
• This is likely a temporary service disruption

**What you can try:**
✅ **Wait and retry**: Most issues resolve within 1-2 minutes
✅ **Simplify your message**: Try shorter, more direct questions
✅ **Start fresh**: Create a new conversation if this one is very long
✅ **Check status**: Visit the service status page for updates

I'll be back online shortly! Thank you for your patience. 🚀""".format(selected_model)
                    logger.error(f"All text models failed for regular chat. Primary: {selected_model}, Tried: {len(models_to_try)} models")
        
            conn.execute(
                'INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)',
                (session_id, "assistant", ai_response)
            )
            conn.execute(
                'UPDATE sessions SET last_updated = ? WHERE id = ?',
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), session_id)
            )
            conn.commit()
            logger.info(f"Assistant response added to session {session_id}")
        
            # Get updated messages for auto-renaming
            updated_messages = conn.execute(
                'SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC', 
                (session_id,)
            ).fetchall()
            
            # Try to auto-rename the session
            auto_rename_session(session_id, updated_messages)
        
        # Ensure we never return null/empty responses
        safe_response = ai_response or "I apologize, but I couldn't generate a response. Please try again."
        
        return jsonify({
            "response": safe_response,
            "message": {"role": "assistant", "content": safe_response},
            "session_id": session_id,
            "success": True
        })
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        
        # Provide more helpful error messages based on the error type
        error_details = handle_model_error(str(e), selected_model if 'selected_model' in locals() else 'AI model')
        
        return jsonify({
            "error": error_details['message'] or "An error occurred",
            "error_type": error_details['type'] or "unknown_error",
            "retry_after": error_details.get('retry_after', 30),
            "technical_details": str(e) if app.debug else None,
            "success": False
        }), 500

def format_messages_for_api(messages_db, current_user_message=None, max_context_messages=20):
    """Format messages for OpenRouter API call with improved context management.
    
    Args:
        messages_db: Database messages for the session
        current_user_message: The current user message to add
        max_context_messages: Maximum number of previous messages to include for context
    
    Returns:
        List of formatted messages for the API
    """
    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Limit context to recent messages to avoid token limits while maintaining conversation flow
    recent_messages = messages_db[-max_context_messages:] if len(messages_db) > max_context_messages else messages_db
    
    # Add conversation summary if we're truncating a long conversation
    if len(messages_db) > max_context_messages:
        summary = generate_conversation_summary(messages_db[:-max_context_messages])
        formatted_messages.append({
            "role": "system", 
            "content": f"Previous conversation summary: {summary}"
        })
    
    for msg in recent_messages:
        if msg['role'] == 'system':
            continue
            
        if msg['has_image'] and msg['image_data'] and msg['image_mime_type']:
            formatted_messages.append({
                "role": msg['role'],
                "content": [
                    {"type": "text", "text": msg['content']},
                    {"type": "image_url", "image_url": {"url": f"data:{msg['image_mime_type']};base64,{msg['image_data']}"}}
                ]
            })
        elif ('has_pdf' in msg.keys() and msg['has_pdf']) and ('pdf_text' in msg.keys() and msg['pdf_text']):
            # For PDF messages, we include the extracted text in the content
            formatted_messages.append({"role": msg['role'], "content": msg['content']})
        else:
            formatted_messages.append({"role": msg['role'], "content": msg['content']})
    
    if current_user_message is not None:
        formatted_messages.append({"role": "user", "content": current_user_message})
        
    return formatted_messages

def generate_conversation_summary(older_messages):
    """Generate a brief summary of older conversation messages.
    
    Args:
        older_messages: List of older messages to summarize
    
    Returns:
        String summary of the conversation
    """
    if not older_messages:
        return "No previous conversation."
    
    # Extract key topics and user queries from older messages
    user_messages = [msg['content'] for msg in older_messages if msg['role'] == 'user']
    assistant_messages = [msg['content'] for msg in older_messages if msg['role'] == 'assistant']
    
    if len(user_messages) == 0:
        return "Previous conversation contained system messages only."
    
    # Create a simple summary
    topics = []
    for msg in user_messages[-3:]:  # Last 3 user messages for context
        if len(msg) > 50:
            topics.append(msg[:50] + "...")
        else:
            topics.append(msg)
    
    summary = f"Earlier, the user discussed: {'; '.join(topics)}"
    return summary[:200]  # Limit summary length

@app.route('/chat_with_file', methods=['POST'])
@requires_valid_session
def chat_with_file():
    """Handle chat messages with image and PDF file uploads."""
    file = request.files.get('file')
    data = request.form.get('data')
    
    if not file or not data:
        return jsonify({"error": "Missing file or data"}), 400
    
    try:
        data = json.loads(data)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON data"}), 400
    
    user_message = data.get('message', '')
    session_id = data.get('session_id', session.get('current_session_id'))
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only images (PNG, JPG, JPEG, GIF) and PDF files are supported."}), 400
        
    filename = sanitize_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400
        
    file_data = file.read()
    
    # Check file size (additional check beyond Flask's MAX_CONTENT_LENGTH)
    if len(file_data) > MAX_CONTENT_LENGTH:
        return jsonify({"error": f"File too large. Maximum size is {MAX_CONTENT_LENGTH // (1024*1024)}MB"}), 400
        
    if len(file_data) == 0:
        return jsonify({"error": "Empty file"}), 400
        
    file_base64 = base64.b64encode(file_data).decode('utf-8')
    file_mime_type = get_mime_type(filename)
    
    # Check if it's a PDF file
    is_pdf = filename.lower().endswith('.pdf')
    
    if is_pdf:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(file_data)
        
        # Check if PDF text extraction was successful
        if pdf_text.startswith("Error:") or pdf_text.startswith("Warning:"):
            return jsonify({"error": f"PDF processing failed: {pdf_text}"}), 400
        
        # Truncate PDF text if it's too long (to avoid token limits)
        max_pdf_length = 8000  # Conservative limit for most models
        if len(pdf_text) > max_pdf_length:
            pdf_text = pdf_text[:max_pdf_length] + "\n\n[Note: PDF content truncated due to length. Only the first portion is shown.]"
            logger.info(f"PDF content truncated from {len(pdf_text)} to {max_pdf_length} characters")
        
        # Format content properly for AI processing
        if user_message:
            content = f"User request: {user_message}\n\nDocument to analyze (PDF: {filename}):\n\n{pdf_text}"
        else:
            content = f"Please analyze and summarize this PDF document (filename: {filename}):\n\n{pdf_text}"
    else:
        content = user_message or f"Uploaded image: {filename}"
    
    try:
        with get_db_connection() as conn:
            session_data = conn.execute('SELECT model FROM sessions WHERE id = ?', (session_id,)).fetchone()
            
            if not session_data:
                conn.execute(
                    'INSERT INTO sessions (id, title, model) VALUES (?, ?, ?)',
                    (session_id, f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}", VISION_MODELS[0])
                )
                conn.commit()
                logger.info(f"New session created with file: {session_id}")
            
            # Use safer query to handle missing columns
            try:
                messages_db = conn.execute(
                    'SELECT role, content, has_image, image_data, image_mime_type, has_pdf, pdf_data, pdf_text FROM messages WHERE session_id = ? ORDER BY timestamp ASC', 
                    (session_id,)
                ).fetchall()
            except sqlite3.OperationalError:
                # Fallback for older database schema
                try:
                    messages_db = conn.execute(
                        'SELECT role, content, has_image, image_data, image_mime_type, 0 as has_pdf, NULL as pdf_data, NULL as pdf_text FROM messages WHERE session_id = ? ORDER BY timestamp ASC', 
                        (session_id,)
                    ).fetchall()
                except sqlite3.OperationalError:
                    # Fallback for even older database schema
                    messages_db = conn.execute(
                        'SELECT role, content, 0 as has_image, NULL as image_data, NULL as image_mime_type, 0 as has_pdf, NULL as pdf_data, NULL as pdf_text FROM messages WHERE session_id = ? ORDER BY timestamp ASC', 
                        (session_id,)
                    ).fetchall()
            
            if is_pdf:
                conn.execute(
                    'INSERT INTO messages (session_id, role, content, has_pdf, pdf_data, pdf_text, image_mime_type) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (session_id, "user", content, 1, file_base64, pdf_text, file_mime_type)
                )
            else:
                conn.execute(
                    'INSERT INTO messages (session_id, role, content, has_image, image_data, image_mime_type) VALUES (?, ?, ?, ?, ?, ?)',
                    (session_id, "user", content, 1, file_base64, file_mime_type)
                )
            conn.execute(
                'UPDATE sessions SET last_updated = ? WHERE id = ?',
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), session_id)
            )
            conn.commit()
            logger.info(f"Image message added to session {session_id}")
        
        if is_pdf:
            # Enhanced PDF processing with better model selection and context management
            pdf_system_prompt = """You are an expert AI document analyst.

📝 Document Analysis:
- Read and comprehend complex documents
- Extract key insights and important information
- Provide structured, well-formatted summaries
- Answer specific questions about document content
- Identify patterns, trends, and conclusions

💡 Response Format:
- Use clear headings and bullet points
- Highlight the most important information
- Provide actionable insights when possible
- Reference specific sections when citing information
- Maintain a professional, concise tone

Always be thorough, accurate, and helpful in your document analysis."""
            
            formatted_messages = [{"role": "system", "content": pdf_system_prompt}]
            
            # Add conversation history but limit it to avoid token overflow
            recent_messages = messages_db[-3:] if len(messages_db) > 3 else messages_db  # Last 3 messages for better context
            for msg in recent_messages:
                if msg['role'] != 'system' and not (msg.get('has_pdf') or msg.get('has_image')):
                    formatted_messages.append({"role": msg['role'], "content": msg['content']})
            
            # Add the current PDF content
            formatted_messages.append({"role": "user", "content": content})
            
            # Enhanced model selection for PDF processing
            preferred_pdf_models = [
                'meta-llama/llama-3.2-90b-vision-instruct:free',
                'meta-llama/llama-4-maverick:free',
                'deepseek/deepseek-chat-v3-0324:free',
                'google/gemini-flash-1.5',
                'meta-llama/llama-3.2-11b-vision-instruct:free'
            ]
            
            # Select best available model for PDF processing
            if session_data and 'model' in session_data and session_data['model']:
                user_preferred_model = session_data['model']
                if user_preferred_model in preferred_pdf_models:
                    models_to_try = [user_preferred_model] + [m for m in preferred_pdf_models if m != user_preferred_model]
                else:
                    models_to_try = [user_preferred_model] + preferred_pdf_models
            else:
                models_to_try = preferred_pdf_models
            
            # Use the enhanced model fallback system
            logger.info(f"Processing PDF with {len(models_to_try)} available models")
            ai_response, used_model = try_models_with_fallback(
                formatted_messages,
                models_to_try,
                max_retries=2,
                model_type="PDF analysis"
            )
            
            if used_model:
                logger.info(f"PDF successfully processed with model: {used_model}")
            
            # Enhanced fallback response for PDF processing
            if not ai_response or not used_model:
                ai_response = """I'm experiencing difficulties processing your PDF document. 📚

**Possible Issues:**
• The PDF might be too large or complex for current processing limits
• Document contains primarily images or scanned text (not machine-readable)
• Temporary service overload or rate limiting
• File format incompatibility

**Solutions to Try:**
✅ **Reduce file size**: Try a smaller PDF or extract specific pages
✅ **Wait and retry**: Service might be temporarily busy
✅ **Text-based PDFs**: Ensure your PDF contains selectable text, not just images
✅ **Manual summary**: Feel free to copy-paste text content for analysis

**Need Help?** Describe what you'd like to know about the document, and I'll guide you through alternative approaches! 🚀"""
                used_model = None
                logger.error("All PDF processing models failed")
        else:
            # For image files, use vision models
            formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            for msg in messages_db:
                if msg['role'] == 'system':
                    continue
                    
                if msg['has_image'] and msg['image_data'] and msg['image_mime_type']:
                    formatted_messages.append({
                        "role": msg['role'],
                        "content": [
                            {"type": "text", "text": msg['content']},
                            {"type": "image_url", "image_url": {"url": f"data:{msg['image_mime_type']};base64,{msg['image_data']}"}}
                        ]
                    })
                else:
                    formatted_messages.append({"role": msg['role'], "content": msg['content']})
            
            formatted_messages.append({
                "role": "user", 
                "content": [
                    {"type": "text", "text": content},
                    {"type": "image_url", "image_url": {"url": f"data:{file_mime_type};base64,{file_base64}"}}
                ]
            })
            
            # Try vision models with fallback
            logger.info(f"Processing image with {len(VISION_MODELS)} available vision models")
            ai_response, used_model = try_models_with_fallback(
                formatted_messages, 
                VISION_MODELS, 
                max_retries=2, 
                model_type="vision"
            )
            
            if used_model:
                logger.info(f"Image successfully processed with vision model: {used_model}")
        
        if ai_response is None:
            # Enhanced fallback response for vision processing
            ai_response = """I'm currently unable to process images due to technical difficulties. 🖼️

**What's happening:**
• Vision processing services are temporarily unavailable
• High demand on image analysis models
• Possible connectivity issues

**What you can do:**
✅ **Describe the image**: Tell me what you see, and I'll help analyze or discuss it
✅ **Try again later**: Image processing usually recovers quickly
✅ **Ask specific questions**: What would you like to know about the image?
✅ **Alternative formats**: Sometimes converting image formats helps

I'm here to help in any way I can! 🚀"""
            logger.error("All vision models failed")
        else:
            logger.info(f"Successfully processed image with model: {used_model}")
        
        with get_db_connection() as conn:
            conn.execute(
                'INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)',
                (session_id, "assistant", ai_response)
            )
            conn.commit()
            logger.info(f"Assistant response to image/PDF added to session {session_id}")
            
            # Get updated messages for auto-renaming
            updated_messages = conn.execute(
                'SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC', 
                (session_id,)
            ).fetchall()
            
            # Try to auto-rename the session
            auto_rename_session(session_id, updated_messages)
        
        # Ensure we never return null/empty responses
        safe_response = ai_response or "I apologize, but I couldn't process your file. Please try again."
        
        return jsonify({
            "response": safe_response,
            "message": {"role": "assistant", "content": safe_response},
            "session_id": session_id,
            "success": True
        })
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        
        # Provide helpful error messages for file processing
        error_msg = str(e).lower()
        if 'timeout' in error_msg:
            user_message = "File processing timed out. Please try with a smaller file or wait a moment and try again."
        elif 'memory' in error_msg or 'size' in error_msg:
            user_message = "File is too large to process. Please try with a smaller file."
        elif 'format' in error_msg or 'invalid' in error_msg:
            user_message = "File format not supported or file is corrupted. Please try a different file."
        else:
            user_message = f"Failed to process file: {str(e)}"
        
        return jsonify({
            "error": user_message or "Failed to process file",
            "error_type": "file_processing_error",
            "technical_details": str(e) if app.debug else None,
            "success": False
        }), 500

@app.route('/get_image/<session_id>/<message_id>', methods=['GET'])
def get_image(session_id, message_id):
    """Serve image data from a message."""
    try:
        with get_db_connection() as conn:
            try:
                message = conn.execute(
                    'SELECT has_image, image_data, image_mime_type FROM messages WHERE session_id = ? AND id = ?', 
                    (session_id, message_id)
                ).fetchone()
            except sqlite3.OperationalError:
                # Fallback for older database schema
                return jsonify({"error": "Image not found"}), 404
        
        if not message or not message['has_image'] or not message['image_data']:
            return jsonify({"error": "Image not found"}), 404
        
        image_data = base64.b64decode(message['image_data'])
        return send_file(BytesIO(image_data), mimetype=message['image_mime_type'])
    except Exception as e:
        logger.error(f"Error retrieving image: {str(e)}")
        return jsonify({"error": f"Failed to retrieve image: {str(e)}"}), 500

@app.route('/get_pdf/<session_id>/<message_id>', methods=['GET'])
def get_pdf(session_id, message_id):
    """Serve PDF data from a message."""
    try:
        with get_db_connection() as conn:
            try:
                message = conn.execute(
                    'SELECT has_pdf, pdf_data, image_mime_type FROM messages WHERE session_id = ? AND id = ?', 
                    (session_id, message_id)
                ).fetchone()
            except sqlite3.OperationalError:
                # Fallback for older database schema
                return jsonify({"error": "PDF not found"}), 404
        
        if not message or not message['has_pdf'] or not message['pdf_data']:
            return jsonify({"error": "PDF not found"}), 404
        
        pdf_data = base64.b64decode(message['pdf_data'])
        return send_file(BytesIO(pdf_data), mimetype='application/pdf')
    except Exception as e:
        logger.error(f"Error retrieving PDF: {str(e)}")
        return jsonify({"error": f"Failed to retrieve PDF: {str(e)}"}), 500

@app.route('/list_sessions', methods=['GET'])
def list_sessions():
    """List all chat sessions."""
    try:
        with get_db_connection() as conn:
            sessions = conn.execute('SELECT * FROM sessions ORDER BY last_updated DESC').fetchall()
        
        return jsonify({
            "sessions": [dict(session) for session in sessions],
            "current_session_id": session.get('current_session_id')
        })
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return jsonify({"error": f"Failed to list sessions: {str(e)}"}), 500

@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors."""
    logger.warning(f"Bad request: {str(error)}")
    return jsonify({"error": "Bad request"}), 400

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit exceeded error."""
    max_size_mb = MAX_CONTENT_LENGTH // (1024 * 1024)
    return jsonify({"error": f"File too large. Maximum file size is {max_size_mb}MB"}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle resource not found error."""
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(429)
def too_many_requests(error):
    """Handle rate limit exceeded error."""
    return jsonify({"error": "Too many requests. Please try again later."}), 429

@app.errorhandler(500)
def server_error(error):
    """Handle internal server error."""
    logger.error(f"Server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(sqlite3.Error)
def database_error(error):
    """Handle database errors with better user feedback."""
    logger.error(f"Database error: {str(error)}")
    
    # Try to provide more specific error messages
    error_msg = str(error).lower()
    if 'locked' in error_msg:
        user_message = "Database is temporarily busy. Please try again in a moment."
    elif 'disk' in error_msg or 'space' in error_msg:
        user_message = "Storage is full. Please contact support."
    elif 'corrupt' in error_msg:
        user_message = "Database corruption detected. Please contact support immediately."
    else:
        user_message = "A database error occurred. Please try again or contact support if the issue persists."
    
    return jsonify({
        "error": user_message,
        "type": "database_error",
        "technical_details": str(error) if app.debug else None
    }), 500

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Handle unexpected errors gracefully."""
    logger.error(f"Unexpected error: {str(error)}", exc_info=True)
    
    return jsonify({
        "error": "An unexpected error occurred. Our team has been notified and will investigate.",
        "type": "unexpected_error",
        "technical_details": str(error) if app.debug else None
    }), 500

if __name__ == '__main__':
    # Application startup cleanup and maintenance
    try:
        cleanup_empty_chats()  # Clean up empty chats
        cleanup_old_sessions(30)  # Clean up old inactive sessions
        logger.info("Application startup cleanup completed")
    except Exception as e:
        logger.warning(f"Startup cleanup failed: {str(e)}")
    
    port = int(os.environ.get('PORT', 9090))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    
    logger.info(f"Starting chat bot server on port {port} with debug={debug}")
    logger.info(f"Default model: {DEFAULT_MODEL}")
    logger.info(f"Available text models: {TEXT_CHAT_MODELS}")
    logger.info(f"Available vision models: {VISION_MODELS}")
    logger.info("Features enabled: Auto-rename, Context Management, File Upload, Enhanced Error Handling")
    
    app.run(debug=debug, host='0.0.0.0', port=port)