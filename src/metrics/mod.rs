pub trait Metric {
    type Input;
    type Result;

    fn update(&mut self, state: &Self::Input);

    fn reset(&mut self);

    fn result(&self) -> Self::Result;
}

impl<M: Metric> Metric for &mut M {
    type Input = M::Input;
    type Result = M::Result;

    fn update(&mut self, state: &Self::Input) {
        (*self).update(state)
    }

    fn reset(&mut self) {
        (*self).reset()
    }

    fn result(&self) -> Self::Result {
        (&**self).result()
    }
}

impl<A: Metric, B: Metric<Input=A::Input>> Metric for (A, B) {
    type Input = A::Input;
    type Result = (A::Result, B::Result);

    fn update(&mut self, state: &Self::Input) {
        self.0.update(state);
        self.1.update(state);
    }

    fn reset(&mut self) {
        self.0.reset();
        self.1.reset();
    }

    fn result(&self) -> Self::Result {
        (self.0.result(), self.1.result())
    }
}

pub struct CombinedMetric<T> {
    inner: T,
}

impl CombinedMetric<()> {
    pub const fn new() -> Self {
        Self { inner: () }
    }
}

impl<T> CombinedMetric<T> {
    pub fn combine<M: Metric>(self, m: &mut M) -> CombinedMetric<(&mut M, T)> {
        CombinedMetric {
            inner: (m, self.inner)
        }
    }
}

impl<M: Metric> Metric for CombinedMetric<M> {
    type Input = M::Input;
    type Result = M::Result;

    fn update(&mut self, state: &Self::Input) {
        self.inner.update(state)
    }

    fn reset(&mut self) {
        self.inner.reset()
    }

    fn result(&self) -> Self::Result {
        self.inner.result()
    }
}
