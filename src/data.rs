use tensor::{BackendProvider, Shape, Tensor};
use void::Void;

pub struct Uninitialized;

pub auto trait Initialized {}

impl ! Initialized for Uninitialized {}

#[const_trait]
pub trait FromRef {
    type Ref<'s>;

    fn from_ref(r: Self::Ref<'_>) -> Self;
}

impl<const N: usize> FromRef for Shape<N> {
    type Ref<'s> = &'s Shape<N>;

    fn from_ref(r: Self::Ref<'_>) -> Self {
        r.clone()
    }
}

impl<const N: usize, const M: usize> FromRef for [Shape<N>; M] {
    type Ref<'s> = [&'s Shape<N>; M];

    fn from_ref(r: Self::Ref<'_>) -> Self {
        r.map(|t| t.clone())
    }
}

impl<A: FromRef, B: FromRef> FromRef for (A, B) {
    type Ref<'s> = (A::Ref<'s>, B::Ref<'s>);

    fn from_ref(r: Self::Ref<'_>) -> Self {
        (FromRef::from_ref(r.0), FromRef::from_ref(r.1))
    }
}

pub trait Package {
    const LEN: usize;
    type Shapes: FromRef;

    fn shape_refs(&self) -> <Self::Shapes as FromRef>::Ref<'_>;
}

impl Package for [Void; 0] {
    const LEN: usize = 0;
    type Shapes = [Shape<0>; 0];

    fn shape_refs(&self) -> <Self::Shapes as FromRef>::Ref<'_> {
        []
    }
}

impl<T, B: BackendProvider, const N: usize> Package for Tensor<T, B, N> {
    const LEN: usize = 1;
    type Shapes = Shape<N>;

    fn shape_refs(&self) -> <Self::Shapes as FromRef>::Ref<'_> {
        self.shape()
    }
}

impl<T, B: BackendProvider, const N: usize, const M: usize> Package for [Tensor<T, B, N>; M] {
    const LEN: usize = M;
    type Shapes = [Shape<N>; M];

    fn shape_refs(&self) -> <Self::Shapes as FromRef>::Ref<'_> {
        self.each_ref().map(|t| t.shape())
    }
}

impl<A: Package, B: Package> Package for (A, B) {
    const LEN: usize = A::LEN + B::LEN;
    type Shapes = (A::Shapes, B::Shapes);

    fn shape_refs(&self) -> <Self::Shapes as FromRef>::Ref<'_> {
        (self.0.shape_refs(), self.1.shape_refs())
    }
}
