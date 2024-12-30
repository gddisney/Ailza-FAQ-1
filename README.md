# FAQ System with Emotional Context and Bert-based Embeddings

## Overview
This project is a web-based FAQ system that combines Bert embeddings and emotional context to provide dynamic and contextually relevant responses. The system uses Actix-web for its web framework and integrates with a custom Bert implementation to generate semantic embeddings for FAQ matching.

### Key Features
- **Bert-based Embeddings**: Utilizes pre-trained GloVe embeddings to compute semantic similarity.
- **Dynamic Emotional Responses**: Adapts to user emotions using pre-defined sentiment templates.
- **FAQ Dataset Management**: Supports adding new questions and answers dynamically.
- **Memory Context**: Generates responses based on recent conversations.
- **Web Interface**: Provides a user-friendly chat interface.

---

## Prerequisites
- **Rust**: Install Rust from [Rust's official website](https://www.rust-lang.org/).
- **Dependencies**:
  - `actix-web`
  - `actix-files`
  - `serde`
  - `serde_json`
  - `ndarray`
  - `rand`
  - `regex`
  - `log`
  - `env_logger`
- **Data Files**:
  - `faq.jsonl`: Contains the FAQ dataset.
  - `emotions.json`: Contains emotion mappings.
  - `word2vex.txt`: Pre-trained GloVe embeddings used through Word2Vec with S2S. Download from [GloVe's official website](https://nlp.stanford.edu/projects/glove/).

---

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Prepare Data Files**:
   - Place `faq.jsonl`, `emotions.json`, and `glove.840B.300d.txt` in the `data/` directory.

3. **Run the Application**:
   ```bash
   cargo run
   ```
   The server starts at `http://127.0.0.1:8080/`.

4. **Access the Chat Interface**:
   Open your browser and navigate to `http://127.0.0.1:8080/`.

---

## FAQ File Format
- The FAQ dataset should be in JSON Lines format (`faq.jsonl`):
  ```json
  {"input": "<PROMPT> How to reset my password?", "response": "<START> To reset your password, follow these steps... <END>"}
  ```

---

## Key Modules
### `FAQ`
Handles all operations related to FAQs, such as initialization, embedding computation, and response generation.

### `FAQSystem`
Coordinates the overall system by integrating Bert, tokenizer, memory, and FAQ management.

### `ask_endpoint`
The API endpoint (`/ask`) processes user input and returns a response.

### `index`
Serves the web-based chat interface.

---

## Development Notes
### Bert Integration
The Bert model leverages GloVe embeddings for initializing word representations. Ensure the `glove.840B.300d.txt` file is properly formatted and accessible.

### Dynamic Response Generation
- Sentiment templates are categorized into positive, negative, and neutral responses.
- Synonyms are applied with a 30% probability for enhanced variety.

### Logging
Enable logging by setting the `RUST_LOG` environment variable:
```bash
RUST_LOG=info cargo run
```

### Memory Management
The system stores the last three interactions to provide context-aware responses.

---

## File Structure
```
.
CDD src
3   CDD main.rs       # Entry point
3   CDD bert.rs       # Bert model implementation
3   @DD ...           # Additional modules
CDD data
3   CDD faq.jsonl     # FAQ dataset
3   CDD emotions.json # Emotion mappings
3   @DD glove.840B.300d.txt # Pre-trained embeddings
CDD Cargo.toml        # Project dependencies
@DD README.md         # Documentation
```

---

## Example API Usage
### Request
```json
POST /ask
{
  "user_input": "How do I reset my password?"
}
```

### Response
```json
{
  "response": "To reset your password, follow these steps..."
}
```

---

## Contributing
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of changes.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- [Actix-web](https://actix.rs/)
- [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)


