use num_traits::Float;

use crate::schedules::LearningRateSchedule;

type F<T: Float + From<f64>> = impl FnMut() -> T;

pub struct ExponentialDecay<T: Float + From<f64>> {
    exp: F<T>,
}

impl<T: Float + From<f64>> ExponentialDecay<T> {
    pub fn new(initial_learning_rate: T, decay_rate: T, decay_steps: u64, staircase: bool) -> Self {
        let f = match staircase {
            true => |decay_rate: T, decay_steps: u64, step: u64| decay_rate.powi((step / decay_steps) as i32),
            false => |decay_rate: T, decay_steps: u64, steps: u64| decay_rate.powf(T::from(steps as f64 / decay_steps as f64)),
        };
        let mut step = 0;
        Self {
            exp: move || initial_learning_rate * f(decay_rate, decay_steps, {
                let t = step;
                step += 1;
                t
            }),
        }
    }
}

impl<T: Float + From<f64>> LearningRateSchedule<T> for ExponentialDecay<T> {
    fn next(&mut self) -> T {
        (self.exp)()
    }
}
