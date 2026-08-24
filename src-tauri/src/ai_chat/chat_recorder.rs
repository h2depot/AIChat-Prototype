use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

pub struct ChatRecorder {
    messages: Vec<ChatMessage>,
    max_chat_tokens: usize,
    used_tokens: usize,
}

impl ChatRecorder {
    pub fn new(max_chat_tokens: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_chat_tokens,
            used_tokens: 0,
        }
    }

    pub fn messages_with_user(&self, content: String) -> Result<Vec<ChatMessage>, String> {
        let mut messages = self.messages.clone();
        messages.push(ChatMessage {
            role: Role::User,
            content,
        });
        Ok(messages)
    }

    pub fn record_exchange(
        &mut self,
        user_content: String,
        assistant_content: String,
        used_tokens: usize,
    ) {
        self.messages.push(ChatMessage {
            role: Role::User,
            content: user_content,
        });
        self.messages.push(ChatMessage {
            role: Role::Assistant,
            content: assistant_content,
        });
        self.used_tokens = used_tokens;
    }

    pub fn clear(&mut self) {
        self.messages.clear();
        self.used_tokens = 0;
    }

    pub fn max_chat_tokens(&self) -> Result<usize, String> {
        Ok(self.max_chat_tokens)
    }
}
