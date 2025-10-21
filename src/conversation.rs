use crate::model::DynamicIntent;
use std::collections::{HashMap, HashSet};

// The output of the recognizer, telling the main logic what it found.
#[derive(Debug, Clone)]
pub enum RecognizedIntent {
    Dynamic(String), // Now carries the name of the matched intent
    Clarification,
    General,
}

#[derive(Debug, Clone)]
pub struct ConversationManager {
    pub last_topic: Option<String>,
    intents: Vec<DynamicIntent>,
}

impl ConversationManager {
    pub fn new(intents: Vec<DynamicIntent>) -> Self {
        Self {
            last_topic: None,
            intents,
        }
    }

    /// Recognizes intent using IDF-weighted keyword scores for more relevance.
    pub fn recognize_intent(
        &mut self,
        text: &str,
        idf_map: &HashMap<String, f32>,
    ) -> RecognizedIntent {
        let lower_text = text.to_lowercase();
        let is_clarification = lower_text.starts_with("and ")
            || lower_text.starts_with("what about")
            || lower_text.starts_with("how about");

        // Rule 1: Handle clarification questions first
        if is_clarification && self.last_topic.is_some() {
            return RecognizedIntent::Clarification;
        }

        // Rule 2: Find the best matching dynamic intent by weighted keyword score
        let text_tokens: HashSet<_> = lower_text.split_whitespace().collect();
        let mut best_match: Option<(&DynamicIntent, f32)> = None;

        for intent in &self.intents {
            let score: f32 = intent
                .keywords
                .iter()
                .filter(|k| text_tokens.contains(k.as_str()))
                // Sum the IDF scores of matched keywords.
                .map(|k| idf_map.get(k).unwrap_or(&1.0))
                .sum();

            if score > 0.0 {
                if let Some((_, best_score)) = best_match {
                    if score > best_score {
                        best_match = Some((intent, score));
                    }
                } else {
                    best_match = Some((intent, score));
                }
            }
        }

        if let Some((best_intent, _)) = best_match {
            self.last_topic = Some(best_intent.name.clone());
            return RecognizedIntent::Dynamic(best_intent.name.clone());
        }

        // Rule 3: Default to general, clearing context
        self.last_topic = None;
        RecognizedIntent::General
    }
}
