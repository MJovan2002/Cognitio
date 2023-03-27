pub mod exponential_decay;
pub mod inverse_time_decay;
pub mod polynomial_decay;

pub trait LearningRateSchedule<T> {
    fn next(&mut self) -> T;
}

impl<T: Clone> LearningRateSchedule<T> for T {
    fn next(&mut self) -> T {
        self.clone()
    }
}
