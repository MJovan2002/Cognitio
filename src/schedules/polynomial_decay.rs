use num_traits::Float;

use crate::schedules::LearningRateSchedule;

pub struct PolynomialDecay<T> {
    initial_learning_rate: T,
    end_learning_rate: T,
    power: T,
    decay_steps: u64,
    f: fn(T, u64, u64) -> T,
    step: u64,
}

impl<T: Float + From<u64>> PolynomialDecay<T> {
    pub fn new(initial_learning_rate: T, end_learning_rate: T, power: T, decay_steps: u64, cycle: bool) -> Self {
        Self {
            initial_learning_rate,
            end_learning_rate,
            power,
            decay_steps,
            f: match cycle {
                true => |power, decay_steps, step| {
                    let t = T::from(step) / T::from(decay_steps);
                    (T::from(1) - t / (t.floor() + T::from(1))).powf(power)
                },
                false => |power, decay_steps, step| (T::from(1) - T::from(step.min(decay_steps)) / T::from(decay_steps)).powf(power),
            },
            step: 0,
        }
    }
}

impl<T: Float> LearningRateSchedule<T> for PolynomialDecay<T> {
    fn next(&mut self) -> T {
        let step = self.step;
        self.step += 1;
        self.end_learning_rate + (self.initial_learning_rate - self.end_learning_rate) * (self.f)(self.power, self.decay_steps, step)
    }
}
