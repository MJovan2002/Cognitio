pub mod mini_batch;
pub mod sgd;
#[cfg(feature = "distributions")]
pub mod noise;
pub mod adagrad;
pub mod adadelta;
pub mod adam;
pub mod adamax;
// pub mod nadam;
pub mod amsgrad;

pub trait Optimizer<G> {
    fn gradients_to_deltas(&mut self, gradients: G) -> Option<G>;
}
