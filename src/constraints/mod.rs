use crate::{
    constraints::none::None as NoneCon,
    data::{Initialized, Uninitialized},
};

pub mod none;
pub mod positive;
// todo: add constraints

pub trait Constraint<T> {
    fn constrain(&self, t: T) -> T;
}

pub trait IntoConstraint<T> {
    type Constraint: Constraint<T>;

    fn into_constraint(self) -> Self::Constraint;
}

impl<T> IntoConstraint<T> for Uninitialized {
    type Constraint = NoneCon<T>;

    fn into_constraint(self) -> Self::Constraint {
        NoneCon::new()
    }
}

impl<T, C: Constraint<T> + Initialized> IntoConstraint<T> for C {
    type Constraint = C;

    fn into_constraint(self) -> Self::Constraint {
        self
    }
}
