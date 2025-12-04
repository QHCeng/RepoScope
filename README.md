# RepoScope

RepoScope is an intelligent GitHub repository explorer that automatically generates a structured **Wiki-style documentation website** and enables **natural-language conversations** grounded in the repository's source code. It provides an interactive way to understand large, unfamiliar codebases by combining static analysis, embedding-based retrieval, and LLM-powered summarization/dialogue.

---

## 🚀 Features

### 📘 1. Automatic Wiki Documentation Generation
RepoScope analyzes an entire GitHub repository and produces a complete Wiki-style documentation tree, including:

- Repository overview  
- Module-level documentation  
- Class, function, and API summaries  
- File-by-file explanations  
- Architecture and dependency diagrams (if enabled)

The generated Wiki aims to provide a clean, readable, and navigable knowledge base for developers.

---

### 💬 2. Code-Aware Conversational Interface
RepoScope enables natural language conversations grounded in the repository's actual source code:

- Ask questions like *“How does the data loader work?”*  
- Query system design decisions  
- Locate functions/implementations  
- Summaries of specific modules or patterns  
- Debug explanations or “where is this used?” lookups  

The conversation engine uses vector embeddings and LLM reasoning to deliver contextually accurate answers.

---

### 🔍 3. Repository Parsing and Indexing
RepoScope performs static analysis on the repository:

- Reads all source files (Python, JS/TS, Go, Rust, etc.)  
- Extracts symbol-level units (classes, functions, methods)  
- Builds embeddings for semantic search  
- Generates a structured knowledge graph of the codebase  
- Produces machine-readable intermediate representations

This index powers both the Wiki generation and conversational features.

---

## 🧠 How It Works

1. Clone or load a GitHub repository  
2. Parse and index files, symbols, and directory structures  
3. Generate embeddings for semantic understanding and retrieval  
4. Populate Wiki templates with AI-generated documentation  
5. Serve a conversation interface backed by the code index  
6. Answer user questions using retrieval-augmented generation  

---
