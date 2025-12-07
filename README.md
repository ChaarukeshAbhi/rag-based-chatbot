# Rag-based chatbot

An intelligent HR chatbot powered by Retrieval-Augmented Generation (RAG) that provides instant, accurate answers to employee queries about company policies, benefits, leave management, and more.
Overview
This chatbot leverages RAG architecture to retrieve relevant information from HR documents and generate contextually accurate responses. It reduces the burden on HR teams by automating responses to common employee questions while maintaining accuracy through document-grounded answers.
Features

Policy Q&A: Instant answers to questions about company policies, procedures, and guidelines
Benefits Information: Detailed information about health insurance, retirement plans, and other benefits
Leave Management: Information about leave policies, accrual, and request procedures
Onboarding Support: Guides new employees through onboarding processes
Document Retrieval: Sources answers from official HR documentation
Context-Aware Responses: Maintains conversation context for natural interactions

Architecture
The system uses a RAG pipeline consisting of:

Document Processing: HR documents are chunked and embedded
Vector Store: Embeddings stored in a vector database for efficient retrieval
Retrieval: Relevant document chunks retrieved based on user queries
Generation: LLM generates responses grounded in retrieved documents
Response: Natural language answer with source attribution

User Query → Embedding → Vector Search → Document Retrieval → LLM → Response
