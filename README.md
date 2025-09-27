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

MidDB is a **middleware database** that bridges the gap between structured storage and semantic embeddings:

- Store structured data (like SQL tables) alongside embeddings for AI memory.  
- Enable both **structured queries** (field-based lookups) and **semantic queries** (nearest-neighbor embedding search).  
- Allow dynamic tables and fields, so memory grows naturally with AI interactions.  
- Serve data via a simple **REST API** for easy integration with AI agents or other services.  
- Provide **persistent storage** of tables and HNSW indices in the `data/` folder.  

---

## MVP (Current Prototype)

The current `MidDB.cpp` includes:

- **In-memory storage** of tables and records.  
- **Async inserts**: Inserts are queued and processed by a dedicated worker thread.  
- **Structured queries** using field filters (`queryField`).  
- **Semantic queries** using approximate nearest neighbor search via HNSW (`queryEmbedding`).  
- **Dynamic tables**: Tables are created automatically if they don’t exist.  
- **Persistent storage**: Records saved in JSON, HNSW indices saved in `.index` files.  
- **REST API** running on `localhost:8080`.  

**Limitations:**

- HNSW index may need optimization for very large datasets.  
- No hybrid queries combining structured + semantic search yet.  
- No automatic embedding generation from AI models.  

---

## Building MidDB
```bash 
git clone https://github.com/nmslib/hnswlib.git
```

### Compile

```bash
g++ -std=c++17 MidDB.cpp -o MidDB -pthread -I./hnswlib -I.
```

### Run
```bash
./MidDB
# Output:
# MidDB (production-ready, async inserts, persistent HNSW) running at http://localhost:8080
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

### Data Storage
-	•	Records → data/<tableName>.json 
-   Stores fields, embeddings, and numeric labels.
-	•	HNSW Index → data/<tableName>.index
- Used for fast approximate nearest-neighbor searches.
-	•	Automatic label mapping is rebuilt from JSON on load.

---

### Architecture
-	1.	Client HTTP Request → /insert or /query* endpoints
-	2.	Insert Queue (async) → Worker thread handles adding records to tables
-	3.	HNSW Index → Approximate nearest-neighbor search for embeddings
-	4.	Persistent Storage → JSON + HNSW index files in data/ folder
-	5.	Query Response → JSON array of matching record IDs

---

### Future Plans
-	•	Hybrid AI Queries: Combine field-based and embedding-based queries.
-	•	Automatic Embeddings: Generate embeddings from AI models on insert.
-	•	Concurrency & Scaling: Multi-threaded inserts, distributed tables.
-	•	Versioned Persistence: Keep history of updates for records.
-	•	Graph Relationships: Connect entities to reason over relationships.
-	•	Cloud/Cluster Support: Scale to multiple nodes or cloud storage.

