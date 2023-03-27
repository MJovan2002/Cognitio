use num_traits::Float;

use crate::schedules::LearningRateSchedule;

pub struct InverseTimeDecay<T> {
    initial_learning_rate: T,
    decay_rate: T,
    decay_steps: u64,
    f: fn(T, u64, u64) -> T,
    step: u64,
}

impl<T: Float + From<i32>> InverseTimeDecay<T> {
    pub fn new(initial_learning_rate: T, decay_rate: T, decay_steps: u64, staircase: bool) -> Self {
        Self {
            initial_learning_rate,
            decay_rate,
            decay_steps,
            f: match staircase {
                true => |decay_rate, decay_steps, step| T::from(1) + decay_rate * T::from((step / decay_steps) as i32),
                false => |decay_rate, decay_steps, step| T::from(1) + decay_rate * T::from(step as i32) / T::from(decay_steps as i32),
            },
            step: 0,
        }
    }
}

impl<T: Float> LearningRateSchedule<T> for InverseTimeDecay<T> {
    fn next(&mut self) -> T {
        let step = self.step;
        self.step += 1;
        self.initial_learning_rate / (self.f)(self.decay_rate, self.decay_steps, step)
    }
}
