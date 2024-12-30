// src/main.rs
use std::io::BufRead;
use actix_files as fs;
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::*;
use std::io::{self, Write};
use serde_json::Value;
use ndarray::{Array2, Array1, Axis, s};
use rand::prelude::*;
use regex::Regex;
use std::sync::Mutex;
use std::fs::read_to_string;
use std::fs::OpenOptions;

// Import the bert module and necessary structs
mod bert;
use bert::{Bert, BertConfig, Tokenizer, EmbeddingFormat};

// Import logging crates
use log::{info, warn, error};
use env_logger;

/// Struct to represent the emotion data loaded from JSON
#[derive(Deserialize, Debug)]
struct EmotionData {
    emotions: HashMap<String, Vec<String>>,
}

/// Struct for the response sent back to the user
#[derive(Serialize)]
struct AskResponse {
    response: String,
}

/// Helper function to capitalize the first letter of a word
fn capitalize_first(word: &str) -> String {
    let mut c = word.chars();
    match c.next() {
        None => "".into(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// Struct representing the FAQ system
struct FAQ {
    questions: Vec<String>,
    answers: Vec<String>,
    question_embeddings: Array2<f32>,
    emotions_map: HashMap<String, String>,
    synonyms: HashMap<String, Vec<String>>,
    templates_positive: Vec<String>,
    templates_negative: Vec<String>,
    templates_neutral: Vec<String>,
}

impl FAQ {
    /// Initialize the FAQ with questions, answers, and other configurations
    fn new(
        questions: Vec<String>,
        answers: Vec<String>,
        bert_model: &Bert,
        tokenizer: &Tokenizer,
        emotions_map: HashMap<String, String>,
        synonyms: HashMap<String, Vec<String>>,
        templates_positive: Vec<String>,
        templates_negative: Vec<String>,
        templates_neutral: Vec<String>,
    ) -> Self {
        // Compute embeddings by averaging all token embeddings for each question
        let mut question_embeddings = Array2::zeros((questions.len(), bert_model.config.hidden_size));
        for (i, question) in questions.iter().enumerate() {
            let token_ids = tokenizer.tokenize(question);
            let embedding = bert_model.forward(&token_ids);
            // Average the embeddings across all tokens
            let avg_embedding = embedding.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(bert_model.config.hidden_size));
            question_embeddings.row_mut(i).assign(&avg_embedding);
        }

        FAQ {
            questions,
            answers,
            question_embeddings,
            emotions_map,
            synonyms,
            templates_positive,
            templates_negative,
            templates_neutral,
        }
    }

    /// Generate a response based on the input embedding and detected emotions
    fn get_response(&self, input_embedding: &Array2<f32>, memory: &[(String, String)], user_input: &str) -> String {
        let detected_emotions = self.detect_emotions(user_input);
        let mut final_response = String::new();

        // Generate empathetic response if emotions are detected
        if !detected_emotions.is_empty() {
            let dynamic_response = self.dynamic_generation(&detected_emotions);
            if !dynamic_response.is_empty() {
                final_response += &self.replace_with_synonyms(dynamic_response);
                final_response += " "; // Add a space between responses
            }
        }

        // Proceed with standard FAQ response
        // Average the input embeddings across all tokens
        let input_vector = input_embedding.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(self.question_embeddings.ncols()));

        let similarity_threshold = 0.8;
        let (best_index, best_score) = self
            .question_embeddings
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                let row_owned = row.to_owned();
                (i, Bert::cosine_similarity(&input_vector, &row_owned))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((0, -1.0));

        if best_score >= similarity_threshold {
            let ans = self.answers[best_index].clone();
            final_response += &self.replace_with_synonyms(ans);
            return final_response;
        }

        if let Some(context) = self.generate_memory_response(memory) {
            final_response += &self.replace_with_synonyms(context);
            return final_response;
        }

        if detected_emotions.is_empty() {
            let neutral_response = self.dynamic_generation(&detected_emotions);
            if !neutral_response.is_empty() {
                final_response += &self.replace_with_synonyms(neutral_response);
                return final_response;
            }
        }

        // If no suitable response is found
        "<RHFL_REQUIRED>".to_string()
    }

    /// Generate a response based on recent memory
    fn generate_memory_response(&self, memory: &[(String, String)]) -> Option<String> {
        if memory.is_empty() {
            return None;
        }
        let mut response = String::from("Based on our recent chat, ");
        for (user_input, bot_response) in memory.iter().rev().take(3) {
            response.push_str(&format!("you mentioned '{}', and I responded '{}'. ", user_input, bot_response));
        }
        response.push_str("Can you clarify or add more details?");
        Some(response)
    }

    /// Generate a dynamic response based on detected emotions
    fn dynamic_generation(&self, detected_emotions: &HashSet<String>) -> String {
        let mut sentiment_label = "neutral";
        if detected_emotions.contains("sadness") || detected_emotions.contains("anger") {
            sentiment_label = "negative";
        } else if detected_emotions.contains("joy") || detected_emotions.contains("love") {
            sentiment_label = "positive";
        }

        let template_list = match sentiment_label {
            "positive" => &self.templates_positive,
            "negative" => &self.templates_negative,
            _ => &self.templates_neutral,
        };

        let mut rng = thread_rng();
        template_list.choose(&mut rng).unwrap_or(&"Let's talk more about that.".into()).clone()
    }

    /// Replace words with their synonyms based on a probability
    fn replace_with_synonyms(&self, text: String) -> String {
        let mut rng = thread_rng();
        let words: Vec<String> = text.split_whitespace().map(|w| w.to_string()).collect();
        let mut result = Vec::new();

        for w in words {
            let lw = w.to_lowercase();
            if let Some(syns) = self.synonyms.get(&lw) {
                if !syns.is_empty() && rng.gen_bool(0.3) { // 30% chance to replace
                    let s = syns.choose(&mut rng).unwrap();
                    result.push(if w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        capitalize_first(s)
                    } else {
                        s.to_string()
                    });
                    continue;
                }
            }
            result.push(w);
        }

        result.join(" ")
    }

    /// Update the FAQ with a new question-answer pair
    fn update_faq(
        &mut self,
        question: String,
        answer: String,
        bert_model: &Bert,
        tokenizer: &Tokenizer,
        faq_file_path: &str,
    ) {
        self.questions.push(question.clone());
        self.answers.push(answer.clone());

        // Compute the embedding by averaging token embeddings
        let token_ids = tokenizer.tokenize(&question);
        let embedding = bert_model.forward(&token_ids);
        let avg_embedding = embedding.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(bert_model.config.hidden_size));

        // Expand the question_embeddings matrix
        let new_shape = (self.question_embeddings.nrows() + 1, self.question_embeddings.ncols());
        let mut updated_embeddings = Array2::zeros(new_shape);
        updated_embeddings
            .slice_mut(s![..-1, ..])
            .assign(&self.question_embeddings);
        updated_embeddings
            .row_mut(self.question_embeddings.nrows())
            .assign(&avg_embedding);
        self.question_embeddings = updated_embeddings;

        // Append the new entry to the FAQ file
        let new_entry = serde_json::json!({
            "input": format!("<PROMPT> {}", question),
            "response": format!("<START> {} <END>", answer),
        });
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(faq_file_path)
            .expect("Failed to open FAQ file");
        writeln!(file, "{}", new_entry).expect("Failed to write to FAQ file");
    }

    /// Detect emotions in the user input based on the emotions_map
    fn detect_emotions(&self, user_input: &str) -> HashSet<String> {
        let words = user_input.split_whitespace().map(|w| w.to_lowercase());
        let mut detected = HashSet::new();
        for w in words {
            if let Some(emotion) = self.emotions_map.get(&w) {
                detected.insert(emotion.clone());
            }
        }
        detected
    }

    /// Generates a comprehensive response using various factors like emotions and memory.
    fn generate_response(&self, input_embedding: &Array2<f32>, memory: &[(String, String)], user_input: &str) -> String {
        let detected_emotions = self.detect_emotions(user_input);
        let mut final_response = String::new();

        // Generate empathetic response if emotions are detected
        if !detected_emotions.is_empty() {
            let dynamic_response = self.dynamic_generation(&detected_emotions);
            if !dynamic_response.is_empty() {
                final_response += &self.replace_with_synonyms(dynamic_response);
                final_response += " "; // Add a space between responses
            }
        }

        // Proceed with standard FAQ response
        let response = self.get_response(input_embedding, memory, user_input);
        if response != "<RHFL_REQUIRED>" {
            final_response += &response;
            return final_response;
        }

        // If no suitable response is found
        "<RHFL_REQUIRED>".to_string()
    }
}

/// Load the FAQ dataset from a JSON Lines file
fn load_faq_dataset(file_path: &str) -> Vec<(String, String)> {
    let file_content = read_to_string(file_path).expect("Failed to read FAQ dataset file");
    file_content
        .lines()
        .filter_map(|line| {
            let json: Value = serde_json::from_str(line).expect("Failed to parse line as JSON");
            let question = json["input"].as_str()?.to_string();
            let answer = json["response"].as_str()?.to_string();
            Some((question, answer))
        })
        .collect()
}

/// Load the emotions map from a JSON file
fn load_emotions_map(file_path: &str) -> HashMap<String, String> {
    let file_content = read_to_string(file_path).expect("Failed to read emotions file");
    let data: EmotionData = serde_json::from_str(&file_content).expect("Failed to parse emotions.json");
    let mut map = HashMap::new();
    for (emotion, words) in data.emotions {
        for w in words {
            map.insert(w.to_lowercase(), emotion.clone());
        }
    }
    map
}

/// Generate the vocabulary from the FAQ dataset
fn generate_vocab(dataset: &[(String, String)]) -> HashMap<String, usize> {
    let mut vocab = std::collections::HashSet::new();
    for (question, answer) in dataset {
        vocab.extend(question.split_whitespace().map(|word| word.to_lowercase()));
        vocab.extend(answer.split_whitespace().map(|word| word.to_lowercase()));
    }

    let mut vocab_map = HashMap::new();
    vocab_map.insert("[PAD]".to_string(), 0);
    vocab_map.insert("[UNK]".to_string(), 1);
    vocab_map.insert("<START>".to_string(), 2);
    vocab_map.insert("<END>".to_string(), 3);

    let mut _index = 4;
    for token in vocab {
        if !vocab_map.contains_key(&token) {
            vocab_map.insert(token, _index);
            _index += 1;
        }
    }

    vocab_map
}

/// Struct representing the entire FAQ system, including BERT model and memory
pub struct FAQSystem {
    faq: FAQ,
    bert_model: Bert,
    tokenizer: Tokenizer,
    faq_file_path: String,
    memory: Vec<(String, String)>,
}

impl FAQSystem {
    /// Initialize the FAQ system with necessary data and models
    pub fn new(
        faq_file_path: &str,
        emotions_file_path: &str,
        word2vec_file_path: &str
    ) -> Self {
        // Load the FAQ dataset and emotions map
        let faq_dataset = load_faq_dataset(faq_file_path);
        let emotions_map = load_emotions_map(emotions_file_path);

        // Generate vocabulary from the dataset
        let vocab_map = generate_vocab(&faq_dataset);
        let vocab_size = vocab_map.len();

        // Initialize the tokenizer with the generated vocabulary
        let tokenizer = Tokenizer::new(vocab_map.clone(), "[UNK]");

        // Define the BERT configuration
        let config = BertConfig {
            hidden_size: 300,               // Must match GloVe 300d
            num_attention_heads: 12,        // Typically a multiple of hidden_size (e.g., 12 for 300 hidden_size)
            intermediate_size: 1200,        // Commonly 4x hidden_size
            num_hidden_layers: 12,           // Number of transformer layers
            vocab_size,
            max_position_embeddings: 512,
            dropout_prob: 0.1,
        };

        // Initialize the BERT model
        let mut bert_model = Bert::new(config.clone());
        bert_model
            .load_word2vec(word2vec_file_path, EmbeddingFormat::GloVe)
            .expect("Failed to load GloVe embeddings");
        bert_model
            .initialize_embeddings_with_word2vec(tokenizer.get_vocab())
            .expect("Failed to initialize embeddings with GloVe vectors");

        // Define synonyms for dynamic response generation
        let mut synonyms = HashMap::new();
        synonyms.insert("talk".to_string(), vec!["discuss".to_string(), "chat".to_string(), "converse".to_string()]);
        synonyms.insert("explore".to_string(), vec!["examine".to_string(), "delve into".to_string(), "investigate".to_string()]);
        synonyms.insert("more".to_string(), vec!["further".to_string(), "additional".to_string(), "extra".to_string()]);
        synonyms.insert("details".to_string(), vec!["info".to_string(), "information".to_string(), "specifics".to_string()]);
        synonyms.insert("sorry".to_string(), vec!["apologetic".to_string(), "regretful".to_string()]);
        synonyms.insert("angry".to_string(), vec!["irked".to_string(), "frustrated".to_string()]);
        synonyms.insert("happy".to_string(), vec!["pleased".to_string(), "delighted".to_string()]);
        synonyms.insert("discuss".to_string(), vec!["examine".to_string(), "review".to_string()]);
        synonyms.insert("further".to_string(), vec!["more deeply".to_string(), "at length".to_string()]);

        // Define response templates based on sentiment
        let templates_positive = vec![
            "That's wonderful! Let's talk more about it.".to_string(),
            "I'm glad to hear that! Can you elaborate?".to_string(),
        ];
        let templates_negative = vec![
            "I'm sorry you're feeling this way. Would you like to discuss it further?".to_string(),
            "That sounds tough. How can I help?".to_string(),
        ];
        let templates_neutral = vec![
            "Let's explore that topic a bit more.".to_string(),
            "I see, can you provide more details?".to_string(),
        ];

        // Clean and prepare the questions and answers
        let mut faq_questions = Vec::new();
        let mut faq_answers = Vec::new();
        for (question, answer) in &faq_dataset {
            let cleaned_question = if question.starts_with("<PROMPT> ") {
                question["<PROMPT> ".len()..].to_string()
            } else {
                question.clone()
            };
            let cleaned_answer = answer
                .replace("<START>", "")
                .replace("<END>", "")
                .trim()
                .to_string();
            faq_questions.push(cleaned_question);
            faq_answers.push(cleaned_answer);
        }

        // Initialize the FAQ struct
        let faq = FAQ::new(
            faq_questions,
            faq_answers,
            &bert_model,
            &tokenizer,
            emotions_map,
            synonyms,
            templates_positive,
            templates_negative,
            templates_neutral
        );

        FAQSystem {
            faq,
            bert_model,
            tokenizer,
            faq_file_path: faq_file_path.to_string(),
            memory: Vec::new(),
        }
    }

    /// Handle user input by generating a response
    pub fn handle_user_input(&mut self, input: &str) -> String {
        info!("Handling user input: {}", input);
        let token_ids = self.tokenizer.tokenize(input);
        let output = self.bert_model.forward(&token_ids);
        let response = self.faq.generate_response(&output, &self.memory, input);
        if response == "<RHFL_REQUIRED>" {
            return "<RHFL_REQUIRED>".to_string();
        } else {
            info!("Generated response: {}", response);
            self.memory.push((input.to_string(), response.clone()));
            response
        }
    }

    /// Handle human feedback by updating the FAQ
    pub fn handle_human_feedback(&mut self, user_input: &str, suggested_response: &str) {
        self.faq.update_faq(user_input.to_string(), suggested_response.to_string(), &self.bert_model, &self.tokenizer, &self.faq_file_path);
    }
}

/// Struct to represent the ask request payload
#[derive(Deserialize)]
struct AskRequest {
    user_input: String,
}

/// Endpoint to handle user questions
#[post("/ask")]
async fn ask_endpoint(req: web::Json<AskRequest>, data: web::Data<Mutex<FAQSystem>>) -> impl Responder {
    let mut system = data.lock().unwrap();
    let response = system.handle_user_input(&req.user_input);
    if response == "<RHFL_REQUIRED>" {
        HttpResponse::Ok().json(AskResponse {
            response: "I am not sure how to respond. Please provide a suitable answer.".to_string()
        })
    } else {
        HttpResponse::Ok().json(AskResponse {
            response
        })
    }
}

/// Endpoint to serve the main chat interface
#[get("/")]
async fn index() -> impl Responder {
    let html = r#"
<!DOCTYPE html>
<html>
<head>
<title>Ailza-FAQ-1 chat</title>
<style>
body {
  font-family: Arial, sans-serif;
  margin: 40px;
  background: #f5f5f5;
}
.chat-container {
  max-width: 600px;
  margin: 0 auto;
  background: #ffffff;
  padding: 20px;
  border-radius: 4px;
}
h1 {
  text-align: center;
}
.message {
  margin-bottom: 10px;
}
.user {
  font-weight: bold;
  color: #333333;
}
.bot {
  font-weight: bold;
  color: #007BFF;
}
input[type=text] {
  width: 80%;
  padding: 10px;
  border: 1px solid #cccccc;
  border-radius: 4px;
}
button {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  background: #007BFF;
  color: #ffffff;
  cursor: pointer;
}
button:hover {
  background: #0056b3;
}
.chat-log {
  height: 400px;
  overflow-y: scroll;
  border: 1px solid #cccccc;
  border-radius: 4px;
  padding: 10px;
  background: #fafafa;
  margin-bottom: 20px;
}
.message-line {
  padding: 5px 0;
}
.message-line span {
  display: inline-block;
  margin-right: 10px;
}
</style>
</head>
<body>
<div class="chat-container">
  <h1>Ailza-FAQ-1 Webchat</h1>
  <div class="chat-log" id="chat-log"></div>
  <input type="text" id="user-input" placeholder="Type your question..." />
  <button onclick="send()">Send</button>
</div>
<script>
async function send() {
    const input = document.getElementById('user-input');
    const userInput = input.value.trim();
    if (userInput === '') return;
    addMessage('You', userInput);
    input.value = '';
    const response = await fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({user_input: userInput})
    });
    const data = await response.json();
    addMessage('Ailza', data.response);
}

function addMessage(sender, text) {
    const log = document.getElementById('chat-log');
    const msg = document.createElement('div');
    msg.className = 'message-line';
    const spanSender = document.createElement('span');
    spanSender.innerText = sender + ':';
    if (sender === 'Ailza') {
        spanSender.className = 'bot';
    } else {
        spanSender.className = 'user';
    }
    const spanText = document.createElement('span');
    spanText.innerText = text;
    msg.appendChild(spanSender);
    msg.appendChild(spanText);
    log.appendChild(msg);
    log.scrollTop = log.scrollHeight;
}
</script>
</body>
</html>
"#;
    HttpResponse::Ok().content_type("text/html; charset=utf-8").body(html)
}

/// Main function to start the Actix-web server
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize the logger
    env_logger::init();
    info!("Starting the FAQ Webchat server...");

    // Define file paths
    let faq_file_path = "data/faq.jsonl";
    let emotions_file_path = "data/emotions.json";
    let word2vec_file_path = "data/glove.840B.300d.txt";

    // Initialize the FAQ system
    let system = FAQSystem::new(faq_file_path, emotions_file_path, word2vec_file_path);
    info!("Initialized the FAQ system.");

    // Share the FAQ system across threads using Mutex and Actix's Data
    let data = web::Data::new(Mutex::new(system));

    // Start the HTTP server
    HttpServer::new(move || {
        App::new()
            .app_data(data.clone())
            .service(index)
            .service(ask_endpoint)
            .service(fs::Files::new("/static", "./static").prefer_utf8(true))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

/// Helper function to load vocabulary from a file
fn load_vocab<P: AsRef<std::path::Path>>(vocab_path: P) -> std::io::Result<HashMap<String, usize>> {
    let file = std::fs::File::open(vocab_path)?;
    let reader = std::io::BufReader::new(file);
    let mut vocab = HashMap::new();
    for (idx, line) in reader.lines().enumerate() {
        let word = line?;
        vocab.insert(word, idx);
    }
    Ok(vocab)
}

