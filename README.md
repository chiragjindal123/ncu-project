# AI Teaching Assistant with Neo4j Knowledge Graph

An intelligent AI Teaching Assistant that combines Neo4j knowledge graphs with Retrieval-Augmented Generation (RAG) to provide personalized learning support through natural language interaction.

Github: [https://github.com/chiragjindal123/ncu-project/tree/ragas](https://github.com/chiragjindal123/ncu-project/tree/ragas)

Demo Video : [https://youtu.be/YunaHJhX82E](https://youtu.be/YunaHJhX82E)

## 🚀 Features

- **Intelligent Conversational AI**: Natural language interaction with context-aware responses
- **Neo4j Knowledge Graph**: Automated concept extraction and relationship discovery
- **RAG Integration**: Hybrid knowledge retrieval combining structured and unstructured data
- **Interactive Quizzes**: Automatic quiz generation with real-time scoring
- **Multi-Model Support**: Gemini API and Ollama integration
- **Document Processing**: Support for PDF, DOCX, and TXT files
- **Semantic Chunking**: Content-aware document processing for better knowledge extraction

## 📋 Prerequisites

- Python 3.10.11
- Docker (for databases)
- Neo4j Desktop
- Gemini API Key

## 🛠️ Installation

### Using uv

You may use `uv` package manager to set different version of python for venv. As this project use python 3.10.11, you can run 

```
uv venv --python 3.10.11
```

### 1. Clone the Repository
```bash
git clone https://github.com/chiragjindal123/ncu-project.git
cd ai-teaching-assistant
```

### 2. Create Python Virtual Environment
```bash
# Create virtual environment with Python 3.12.0
python3.10.11 -m venv ai_ta_env

# Activate virtual environment
# On Windows:
ai_ta_env\Scripts\activate
# On macOS/Linux:
source ai_ta_env/bin/activate

# Verify Python version
python --version  # Should show Python 3.10.11
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Databases

#### PostgreSQL with pgvector (for RAG embeddings)
```bash
docker run --name ai_pgvector \
  -e POSTGRES_USER=aiuser \
  -e POSTGRES_PASSWORD=aipassword \
  -e POSTGRES_DB=aidb \
  -p 5432:5432 \
  -d ankane/pgvector
```

#### Infinity Embeddings Server (for vector generation)
```bash
docker run --name infinity_embeddings \
  -p 8080:8080 \
  -d michaelf34/infinity:latest
```

#### Neo4j Database Setup
You can use Neo4j Desktop or Docker.

1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new project and database
3. Set password (e.g., "password123")
4. Start the database
5. Note the connection URI (default: neo4j://localhost:7687)

Using docker (without persisting data between restart)

```bash
docker run \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    neo4j:2025.10.1
```

Then open http://localhost:7474/ (the Neo4j’s Browser interface) in a web browser to set a new password (default neo4j/neo4j at first login).

### 5. Environment Configuration

Create a `.env` file in the project root:
```env
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent
OPENAI_API_KEY=your_key

# Neo4j Configuration
NEO4J_PASSWORD=password123

# Database Configuration (if different from defaults)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=aiuser
POSTGRES_PASSWORD=aipassword
POSTGRES_DB=aidb

# Infinity Embeddings
INFINITY_API_URL=http://localhost:8080
```

### 6. Initialize Database Tables

Run the application once to create necessary tables:
```bash
python app.py
```

The system will automatically create required PostgreSQL tables on first run.

## 🚀 Usage

### 1. Start All Services
```bash
# Start PostgreSQL (if not running)
docker start ai_pgvector

# Start Infinity Embeddings (if not running)
docker start infinity_embeddings

# Start Neo4j (from Neo4j Desktop)
# Ensure your Neo4j database is running

# Activate virtual environment (if not active)
source ai_ta_env/bin/activate  # macOS/Linux
# or
ai_ta_env\Scripts\activate     # Windows

# Start Flask application
python app.py
```

### 2. Access the Application
Open your browser and navigate to: `http://localhost:5000`

### 3. Upload Documents
- Click "Choose File" and select a PDF, DOCX, or TXT file
- The system will:
  - Extract text content
  - Create semantic chunks
  - Build knowledge graph in Neo4j
  - Generate vector embeddings for RAG

### 4. Interactive Learning
- **Ask Questions**: Type natural language questions
- **Generate Quizzes**: Ask "Create a quiz about [topic]"
- **Review Concepts**: Ask "Explain [concept]"
- **Practice Problems**: Ask "Give me practice exercises on [topic]"

### 5. View Knowledge Graph
Access Neo4j Browser at `http://localhost:7474` to visualize the knowledge graph:
```cypher
// View all nodes and relationships
MATCH (n) RETURN n LIMIT 25

// See topic hierarchy
MATCH (t:Topic)-[:HAS_SUBTOPIC]->(s:Subtopic)
RETURN t.name, collect(s.name) as subtopics

// Find concept relationships
MATCH (t1:Topic)-[r:RELATED_CONCEPT]->(t2:Topic)
RETURN t1.name, type(r), t2.name
```

## 📁 Project Structure

```
ai-teaching-assistant/
├── .env                    # Environment variables
├── .gitignore             # Git ignore file
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── app.py                 # Main Flask application
├── knowledge_extractor.py # LLM-based knowledge extraction
├── knowledge_graph.py     # Neo4j graph operations
├── rag_utils.py          # RAG utilities and database operations
└── templates/
    └── index.html        # Web interface
```

## 🔧 Configuration

### Model Configuration
- **Embeddings**: sentence-transformers/all-mpnet-base-v2
- **LLM**: Gemini 2.0 Flash (via API)
- **Local LLM**: Ollama (optional)

### Performance Tuning
- **Chunk Size**: 1000 characters (RAG)
- **Semantic Chunk Size**: 900 characters (Knowledge extraction)
- **Overlap**: 200 characters
- **Similarity Threshold**: 0.65

## 🐛 Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   ```bash
   # Check if Neo4j is running
   # Verify password in .env file
   # Ensure port 7687 is not blocked
   ```

2. **PostgreSQL Connection Error**
   ```bash
   # Check Docker container status
   docker ps
   
   # Restart if needed
   docker restart ai_pgvector
   ```

3. **Gemini API Errors**
   ```bash
   # Verify API key in .env
   # Check internet connectivity
   # Monitor API quota limits
   ```

4. **Embedding Server Issues**
   ```bash
   # Check Infinity server status
   docker logs infinity_embeddings
   
   # Restart if needed
   docker restart infinity_embeddings
   ```

### Performance Issues
- Reduce document size if processing is slow
- Check system resources (CPU/Memory usage)
- Monitor database performance in Neo4j Browser

## 📊 Monitoring and Analytics

The system includes built-in analytics:
- Response time monitoring
- Knowledge extraction accuracy
- Quiz effectiveness metrics
- Graph connectivity analysis

Access metrics at: `http://localhost:5000/project_evaluation`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Neo4j for graph database technology
- Google for Gemini API
- Sentence Transformers for embedding models
- pgvector for PostgreSQL vector operations

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs in terminal/console
3. Check database connectivity
4. Verify API keys and environment variables

---
