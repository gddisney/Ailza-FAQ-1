use crate::model::DynamicIntent;

// The output of the recognizer, telling the main logic what it found.
#[derive(Debug, Clone)]
pub enum RecognizedIntent {
    Dynamic, // The name of the matched DynamicIntent
    Clarification,
    General,
}

#[derive(Debug, Clone)]
pub struct ConversationManager {
    pub last_topic: Option<String>,
    intents: Vec<DynamicIntent>,
}

impl ConversationManager {
    // The manager is now created with the list of dynamic intents
    pub fn new(intents: Vec<DynamicIntent>) -> Self {
        Self {
            last_topic: None,
            intents,
        }
    }

    pub fn recognize_intent(&mut self, text: &str) -> RecognizedIntent {
        let lower_text = text.to_lowercase();
        let is_clarification = lower_text.starts_with("and ")
            || lower_text.starts_with("what about")
            || lower_text.starts_with("how about");

        // Rule 1: Handle clarification questions first if there's a topic in memory
        if is_clarification && self.last_topic.is_some() {
            return RecognizedIntent::Clarification;
        }

        // Rule 2: Find the best matching dynamic intent by keyword count.
        // This is more robust than returning on the first partial match.
        let mut best_match: Option<(&DynamicIntent, usize)> = None;

        for intent in &self.intents {
            let match_count = intent
                .keywords
                .iter()
                .filter(|k| lower_text.contains(*k))
                .count();

            if match_count > 0 {
                if let Some((_, best_count)) = best_match {
                    if match_count > best_count {
                        best_match = Some((intent, match_count));
                    }
                } else {
                    best_match = Some((intent, match_count));
                }
            }
        }

        if let Some((best_intent, _)) = best_match {
            // Save the matched intent name as the last topic
            self.last_topic = Some(best_intent.name.clone());
            return RecognizedIntent::Dynamic;
        }


        // Rule 3: If no specific intent is found, it's a general question.
        // Clear the topic since the context has likely changed.
        self.last_topic = None;
        RecognizedIntent::General
    }
}
