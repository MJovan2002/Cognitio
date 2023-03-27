use crate::{
    datasets::Dataset,
    model::Model,
    optimizers::Optimizer,
    losses::Loss,
};
use crate::layers::Layer;

pub struct Trainer<'m, M: Layer, O: Optimizer<M::Internal>> {
    model: &'m mut Model<M>,
    optimizer: O,
}

impl<'m, M: Layer, O: Optimizer<M::Internal>> Trainer<'m, M, O> {
    pub(crate) fn new(model: &'m mut Model<M>, optimizer: O) -> Self {
        Self { model, optimizer }
    }

    pub fn train<DS: Dataset<Input=M::Input>, L: Loss<M::Output, M::ReverseOutput>>(
        &mut self,
        epochs: usize,
        dataset: &DS,
        loss: L,
        label_to_output: fn(DS::Label) -> M::Output,
    ) { // todo: add metrics and callbacks, integrate losses into model, return history
        for _ in 0..epochs {
            for (input, expected) in dataset.get_training_iter() {
                let (predicted, computation) = self.model.back_propagate(input);
                let derivatives = loss.derive(&predicted, &label_to_output(expected));
                let (_, gradients) = computation(derivatives);
                if let Some(deltas) = self.optimizer.gradients_to_deltas(gradients) {
                    self.model.update(&deltas);
                }
            }
        }
    }
}
