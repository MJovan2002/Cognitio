pub mod mnist;
// todo: add datasets

pub trait Dataset {
    type Input;
    type Label;
    type Iter: Iterator<Item = (Self::Input, Self::Label)>;

    fn get_training_iter(&self) -> Self::Iter;

    fn get_testing_iter(&self) -> Self::Iter;
}
