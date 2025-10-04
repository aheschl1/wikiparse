# WikiParse

A Python library for processing Wikipedia dumps, building semantic search indices, and visualizing document link networks using ColBERT embeddings and FAISS.

## Overview

WikiParse provides a complete pipeline for:
- Parsing Wikipedia XML dump files into structured documents
- Creating chunked text embeddings using ColBERT v2.0
- Building efficient FAISS-based search indices with sharding
- Visualizing Wikipedia link networks as interactive graphs
- Querying documents using semantic similarity search

## Installation

```bash
pip install -e .
```

### Dependencies

- PyTorch
- Transformers (4.56.2)
- Pydantic (2.11.9)
- SQLAlchemy
- FAISS
- NetworkX
- Plotly
- Click

## Quick Start

### 1. Parse Wikipedia Dump

Extract and chunk Wikipedia articles from dump files:

```bash
python -m wikindex.wiki.wiki extracted/ --extract --db wikindex.db --cp 8
```

### 2. Generate Embeddings

Create ColBERT embeddings for all document chunks:

```bash
python -m wikindex.wiki.wiki extracted/ --embed --db wikindex.db --cp 8
```

### 3. Build Search Index

Create a sharded FAISS index for efficient search:

```bash
python -m wikindex.data.index index_folder/ extracted/ --db-url wikindex.db --train
```

### 4. Visualize Link Network

Generate an interactive HTML visualization of document links:

```bash
python wikindex/gen_wikigraph.py
```

## Architecture

### Core Components

- **WikiDoc**: Document model with automatic text chunking
- **WikiDataset**: Dataset interface for Wikipedia documents
- **ColBert**: Token-level embedding encoder with projection layers
- **FaissIndex**: Scalable search index with automatic sharding
- **WikiGraph**: Network analysis and visualization tools

### Database Schema

- **documents**: Article metadata (id, title, url)
- **chunks**: Text segments with embeddings
- **document_links**: Link relationships between articles
- **files**: Processing status tracking

### Embedding Strategy

- Uses ColBERT v2.0 for contextual token embeddings
- 128-dimensional projections with L2 normalization
- Punctuation filtering for document embeddings
- Configurable chunking with sentence boundary preservation

## Configuration

Default settings in `wikindex/config.py`:

```python
max_tokens: 512          # Maximum tokens per chunk
device: 'cuda'           # Computation device
max_batch_size: 1050     # Embedding batch size
top_k: 10               # Search result count
projected_dim: 128       # Embedding dimensions
```

## Command Line Interface

### Wiki Processing
```bash
python -m wikindex.wiki.wiki ROOT [OPTIONS]
  --db-name TEXT         Database file or URL
  --consumer-processes   Number of worker processes
  --extract             Parse Wikipedia dump files
  --embed               Generate embeddings
```

### Index Building
```bash
python -m wikindex.data.index ROOT EMBEDDINGS_PATH [OPTIONS]
  --db-url TEXT         Database URL
  --train               Train new quantizer
```

## API Usage

### Basic Search
```python
from wikindex.wiki.sqlite import Client
from wikindex.wiki.wiki import WikiDataset
from wikindex.custom_colbert.model import ColBert
from wikindex.data.index import FaissIndex

# Load dataset
with Client("sqlite:///wikindex.db") as client:
    encoder = ColBert(config=config)
    dataset = WikiDataset.from_sqlite(client, encoder, config)
    
    # Load index
    index = FaissIndex.load(dataset, encoder, scorer, 
                           embeddings_path, index_path, config)
    
    # Search
    scores, results = index.search("machine learning", top_k=5)
```

### Graph Analysis
```python
from wikindex.wiki.wikigraph import visualize_document_links

# Create interactive visualization
fig = visualize_document_links(
    db_url="sqlite:///wikindex.db",
    layout='spring',
    node_size=8,
    show_labels=False
)
fig.write_html("wikipedia_graph.html")
```

## Performance

### Scaling
- Supports multi-process parsing and embedding generation
- Automatic FAISS index sharding for large datasets
- Configurable batch sizes for memory management
- Progress tracking with detailed logging

### Optimization
- IVF-PQ quantization for memory efficiency
- Token-level MaxSim scoring for relevance
- Lazy loading of embeddings during search
- Database connection pooling

## Testing

Run the test suite:
```bash
pytest wikindex/tests/
```

Test coverage includes:
- Document parsing and chunking
- HTML content cleaning
- Embedding generation
- Database operations

