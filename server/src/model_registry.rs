use model::Agent;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};

use crate::protocol::ModelInfo;

/// An entry in the model registry: the agent and its broadcast channel.
pub struct RegisteredModel {
    pub agent: Arc<dyn Agent + Send + Sync>,
    pub sender: broadcast::Sender<String>,
}

struct RegistryInner {
    models: HashMap<String, RegisteredModel>,
    /// Ordered list of model names for consistent iteration.
    order: Vec<String>,
}

/// Registry of all available models. Each model runs its own agent game loop
/// and broadcasts state to watchers. Supports runtime additions via interior
/// mutability for hot model loading.
pub struct ModelRegistry {
    inner: RwLock<RegistryInner>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(RegistryInner {
                models: HashMap::new(),
                order: Vec::new(),
            }),
        }
    }

    /// Registers a model. Returns the broadcast sender for the new model's
    /// game loop, or None if a model with this name is already registered.
    pub async fn register(
        &self,
        agent: Arc<dyn Agent + Send + Sync>,
    ) -> Option<broadcast::Sender<String>> {
        let name = agent.name().to_string();
        let mut inner = self.inner.write().await;

        if inner.models.contains_key(&name) {
            return None;
        }

        let (sender, _) = broadcast::channel(16);
        let sender_clone = sender.clone();
        inner.order.push(name.clone());
        inner
            .models
            .insert(name, RegisteredModel { agent, sender });
        Some(sender_clone)
    }

    /// Subscribes to a model's broadcast channel by name.
    pub async fn subscribe(&self, name: &str) -> Option<broadcast::Receiver<String>> {
        let inner = self.inner.read().await;
        inner.models.get(name).map(|model| model.sender.subscribe())
    }

    /// Returns info about all registered models, in registration order.
    pub async fn list(&self) -> Vec<ModelInfo> {
        let inner = self.inner.read().await;
        inner
            .order
            .iter()
            .filter_map(|name| {
                inner.models.get(name).map(|model| ModelInfo {
                    name: model.agent.name().to_string(),
                    description: model.agent.description().to_string(),
                })
            })
            .collect()
    }

    /// Returns the name of the first registered model (default for new connections).
    pub async fn default_model(&self) -> Option<String> {
        let inner = self.inner.read().await;
        inner.order.first().cloned()
    }

    /// Returns a snapshot of all registered models' agents and senders,
    /// suitable for spawning game loops.
    pub async fn snapshot(
        &self,
    ) -> Vec<(String, Arc<dyn Agent + Send + Sync>, broadcast::Sender<String>)> {
        let inner = self.inner.read().await;
        inner
            .order
            .iter()
            .filter_map(|name| {
                inner.models.get(name).map(|model| {
                    (name.clone(), model.agent.clone(), model.sender.clone())
                })
            })
            .collect()
    }
}
