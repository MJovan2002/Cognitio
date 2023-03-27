pub mod square;
// todo: add losses

pub trait Loss<T, U> {
    // fn loss(&self, predicted: &C, expected: &C) -> T;

    fn derive(&self, predicted: &T, expected: &T) -> U;
}
