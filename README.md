# MidDB

**MidDB** is a lightweight, AI-aware hybrid database prototype written in C++. It combines structured data storage with vector embeddings for semantic memory, enabling both traditional queries and AI-driven similarity searches.

---

## Problem

Modern AI agents and LLM-powered applications often require **dynamic memory** that can store structured information and semantic embeddings together. Existing solutions fall short in several ways:

- SQL databases are great for structured data but cannot handle semantic search efficiently.  
- Vector databases (like FAISS or Pinecone) handle embeddings but lack structured query capabilities.  
- AI agents need a **dynamic, hybrid memory** to store, retrieve, and reason over data seamlessly.  

---

## Idea

MidDB aims to be a **middleware database** that bridges the gap between structured storage and semantic embeddings:

- Store structured data (like SQL tables) alongside embeddings for AI memory.  
- Enable both **structured queries** (field-based lookups) and **semantic queries** (nearest-neighbor embedding search).  
- Allow dynamic tables and fields, so memory can grow naturally with AI interactions.  
- Serve data via a simple **REST API** for easy integration with AI agents or other services.  

---

## MVP (Current Prototype)

The current `MidDB.cpp` is a working prototype with the following features:

- **In-memory storage** of tables and records.  
- **Structured queries** using field filters (`queryField`).  
- **Semantic queries** using Euclidean distance on embeddings (`queryEmbedding`).  
- **Dynamic tables**: Tables are created automatically if they don’t exist.  
- **REST API** for insertions and queries, running on `localhost:8080`.  
- **JSON wrapper** for optional persistence (can be extended to save/load data).  

**Limitations of MVP:**

- No persistent storage (data is lost when the server restarts).  
- Linear search for semantic queries (not optimized for large datasets).  
- Single-threaded, no concurrency handling.  
- No hybrid queries combining structured + semantic search.  
- No automatic embedding generation from AI models.  

---

## Building MidDB

The roadmap to a full-fledged MidDB includes:

1. **Persistence**
   - Save/load database in a JSON format (prototype).  
   - Optionally add binary formats or integrate with SQLite/PostgreSQL.  

2. **Optimized Semantic Search**
   - Implement HNSW, FAISS, or other ANN algorithms for fast vector search.  

3. **Hybrid Queries**
   - Combine structured and semantic queries in one request.  
   - Example: “Find orders by buyers named Alice that are semantically closest to this vector.”  

4. **Dynamic AI Memory**
   - Integrate LLMs to generate embeddings automatically when inserting new records.  
   - Support versioned or time-based memory for agents.  

5. **Concurrency & Scaling**
   - Multi-threaded query handling.  
   - Optionally split tables across processes or nodes for scale.  

6. **Graph Relationships**
   - Connect entities with edges to allow reasoning over relationships.  

---

## How to Use the MVP

### Compile

```bash
g++ -std=c++17 MidDB.cpp -o MidDB -pthread
```

### Run
```bash
./MidDB
Server runs at http://localhost:8080.
```

---

### Insert a Record
```bash
curl -X POST http://localhost:8080/insert \
-H "Content-Type: application/json" \
-d '{
  "table": "users",
  "id": "user1",
  "fields": {"name": "Alice", "email": "alice@example.com"},
  "embedding": [0.1, 0.5, 0.2]
}'
```

---

### Structured Query
```bash
curl "http://localhost:8080/queryField/users?field=name&value=Alice"
# Output: ["user1"]
```

---

### Semantic Query
```bash
curl -X POST http://localhost:8080/queryEmbedding/users \
-H "Content-Type: application/json" \
-d '{
  "embedding": [0.1, 0.5, 0.2],
  "topK": 1
}'
# Output: ["user1"]
```

---

### Future Plans
-	•	AI integration: Auto-generate embeddings for inserted data.
-	•	Persistent storage: MidDB files with versioning.
-	•	Hybrid AI queries: Combine structured and semantic search for smarter retrieval.
-	•	Scalability: Multi-threading and optimized indexing for large datasets.

---

MidDB is a foundation for AI-aware memory systems, bridging structured databases and vector search into a single, extensible platform.

