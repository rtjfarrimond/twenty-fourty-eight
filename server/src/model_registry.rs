use model::Agent;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;

use crate::protocol::ModelInfo;

/// An entry in the model registry: the agent and its broadcast channel.
pub struct RegisteredModel {
    pub agent: Arc<dyn Agent + Send + Sync>,
    pub sender: broadcast::Sender<String>,
}

/// Registry of all available models. Each model runs its own agent game loop
/// and broadcasts state to watchers.
pub struct ModelRegistry {
    models: HashMap<String, RegisteredModel>,
    /// Ordered list of model names for consistent iteration.
    order: Vec<String>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Registers a model. The broadcast channel is created internally.
    pub fn register(&mut self, agent: Arc<dyn Agent + Send + Sync>) {
        let name = agent.name().to_string();
        let (sender, _) = broadcast::channel(16);
        self.order.push(name.clone());
        self.models.insert(name, RegisteredModel { agent, sender });
    }

    /// Returns the registered model by name.
    pub fn get(&self, name: &str) -> Option<&RegisteredModel> {
        self.models.get(name)
    }

    /// Returns info about all registered models, in registration order.
    pub fn list(&self) -> Vec<ModelInfo> {
        self.order
            .iter()
            .filter_map(|name| {
                self.models.get(name).map(|model| ModelInfo {
                    name: model.agent.name().to_string(),
                    description: model.agent.description().to_string(),
                })
            })
            .collect()
    }

    /// Returns the name of the first registered model (default for new connections).
    pub fn default_model(&self) -> Option<&str> {
        self.order.first().map(|s| s.as_str())
    }

    /// Iterates over all registered models.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &RegisteredModel)> {
        self.order
            .iter()
            .filter_map(move |name| self.models.get(name).map(|model| (name, model)))
    }
}
