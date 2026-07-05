use crate::{
    Muscle, NervousSystem,
};
use genetic_ops::prelude::*;
use bevy::prelude::*;
use petgraph::{algo::toposort, prelude::*};
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::f32::consts::TAU;

const NEURON_VARIANT_COUNT: usize = 15;
#[derive(Serialize, Deserialize, Clone, Debug, Copy, PartialEq)]
pub enum Neuron {
    Sensor,
    Muscle,
    Sin {
        amp: f32,
        freq: f32,
        phase: f32,
    },
    Complement,
    Const(f32),
    Scale(f32),
    Mult,
    /// Divide first input by second
    Div,
    /// Sums inputs
    Sum,
    /// Subtracts first input from second.
    Diff,
    /// Outputs difference between current and previous input, scaled to units of change 0.1 sec with evolvable direction flag
    Deriv {
        dir: bool,
    },
    /// Outputs 1 if >= .0; otherwise outputs 0.
    Threshold(f32),
    /// If first input is >= .0, output second input; otherwise outputs 0.
    Switch(f32),
    /// Applies an evolvable delay to input signal.
    Delay(u8),
    /// Outputs the absolute difference of input units.
    AbsDiff,
}

impl From<NVec4> for Neuron {
    fn from(v: NVec4) -> Neuron {
        v.0.into()
    }
}

impl From<Vec4> for Neuron {
    fn from(v: Vec4) -> Neuron {
        use Neuron::*;
        match (v.x * NEURON_VARIANT_COUNT as f32) as usize % NEURON_VARIANT_COUNT {
            0 => Sensor,
            1 => Muscle,
            2 => Sin {
                amp: v.y,
                freq: v.z,
                phase: v.w,
            },
            3 => Complement,
            4 => Const(v.y),
            5 => Scale(v.y),
            6 => Mult,
            7 => Div,
            8 => Sum,
            9 => Diff,
            10 => Deriv { dir: v.z > 0.5 },
            11 => Threshold(v.y),
            12 => Switch(v.y),
            13 => Delay((v.w * 5.0) as u8),
            14 => AbsDiff,
            _ => {
                todo!();
            }
        }
    }
}

impl From<Neuron> for NVec4 {
    fn from(n: Neuron) -> NVec4 {
        NVec4(n.into())
    }
}

/// Try to avoid using this since it converts to a lossy representation.
impl From<Neuron> for Vec4 {
    fn from(n: Neuron) -> Vec4 {
        use Neuron::*;
        let mut v = Vec4::ZERO;
        match n {
            Sensor => v.x = 0.0,
            Muscle => v.x = 1.0,
            Sin { amp, freq, phase } => {
                v.x = 2.0;
                v.y = amp;
                v.z = freq;
                v.w = phase;
            }
            Complement => v.x = 3.0,
            Const(c) => {
                v.x = 4.0;
                v.y = c;
            }
            Scale(s) => {
                v.x = 5.0;
                v.y = s;
            }
            Mult => v.x = 6.0,
            Div => v.x = 7.0,
            Sum => v.x = 8.0,
            Diff => v.x = 9.0,
            Deriv { dir } => {
                v.x = 10.0;
                v.y = if dir { 1.0 } else { 0.0 };
            }
            Threshold(t) => {
                v.x = 11.0;
                v.y = t;
            }
            Switch(t) => {
                v.x = 12.0;
                v.y = t;
            }
            Delay(d) => {
                v.x = 13.0;
                v.w = d as f32 / 5.0;
            }
            AbsDiff => v.x = 14.0,
        }
        v.x /= NEURON_VARIANT_COUNT as f32;
        v
    }
}

#[derive(Debug, Deref, DerefMut, PartialEq, Clone, Copy)]
pub struct NVec4(pub Vec4);

impl From<Vec4> for NVec4 {
    fn from(v: Vec4) -> NVec4 {
        NVec4(v)
    }
}

impl NVec4 {
    pub fn gen<R>(rnd: &mut R) -> Self
    where
        R: Rng,
    {
        let g = uniform_generator(0.0, 1.0);
        NVec4(Vec4::new(
            g.gen(rnd),
            g.gen(rnd),
            g.gen(rnd),
            g.gen(rnd),
        ))
    }

    pub fn mutate_all<R>(gene: &mut Self, rnd: &mut R) -> u32
    where
        R: Rng,
    {
        let m = uniform_mutator(0.0, 0.1);
        let mut count = 0;
        count += m.mutate(&mut gene.0.x, rnd);
        count += m.mutate(&mut gene.0.y, rnd);
        count += m.mutate(&mut gene.0.z, rnd);
        count += m.mutate(&mut gene.0.w, rnd);
        count
    }

    pub fn mutate_one<R>(gene: &mut Self, rnd: &mut R) -> u32
    where
        R: Rng,
    {
        let m = uniform_mutator(0.0, 0.1);
        match rnd.random_range(0..4) {
            0 => m.mutate(&mut gene.0.x, rnd),
            1 => m.mutate(&mut gene.0.y, rnd),
            2 => m.mutate(&mut gene.0.z, rnd),
            3 => m.mutate(&mut gene.0.w, rnd),
            _ => unreachable!(),
        }
    }
}

pub struct Context {
    time: f32,
}

pub fn plugin(app: &mut App) {
    app.add_systems(Update, bitbrain_update);
}

pub fn bitbrain_update(
    time: Res<Time>,
    mut nervous_systems: Query<(&NervousSystem, &mut BitBrain)>,
    mut muscles_query: Query<&mut Muscle>,
) {
    let ctx = Context {
        time: time.elapsed_secs(),
    };
    for (nervous_system, mut brain) in &mut nervous_systems {
        let _sensors = &nervous_system.sensors;
        // TODO: Update the sensor values.
        brain.eval(&ctx);
        let muscles = &nervous_system.muscles;
        for (i, muscle) in muscles.iter().enumerate() {
            if let Ok(mut muscle) = muscles_query.get_mut(*muscle) {
                if let Some(v) = brain.read_muscle(i) {
                    muscle.value = *v;
                }
            } else {
                break;
            }
        }
    }
}

impl Neuron {
    fn eval(
        &self,
        context: &Context,
        state: f32,
        inputs: &[f32],
        aux: Option<&mut VecDeque<f32>>,
    ) -> f32 {
        use Neuron::*;
        match self {
            Sensor => state,
            Muscle => inputs.iter().sum(),
            Sin { amp, freq, phase } => (context.time * freq * TAU + phase).sin() * amp / 2.0 + 0.5,
            Complement => 1.0 - inputs.iter().sum::<f32>(),
            Const(c) => *c,
            Scale(s) => s * inputs.iter().sum::<f32>(),
            Sum => inputs.iter().sum::<f32>(),
            Mult => inputs.iter().product(),
            Div => inputs
                .first()
                .map(|f| f / inputs.iter().skip(1).sum::<f32>())
                .unwrap_or(0.0),
            Diff => inputs
                .first()
                .map(|f| f - inputs.iter().skip(1).sum::<f32>())
                .unwrap_or(0.0),
            Deriv { dir: _ } => {
                // todo!("Deriv")
                0.0
            }
            Threshold(t) => inputs
                .first()
                .and_then(|f| (*f >= *t).then_some(1.0))
                .unwrap_or(0.0),
            Switch(t) => inputs
                .first()
                .and_then(|f| (*f >= *t).then_some(inputs.iter().skip(1).sum::<f32>()))
                .unwrap_or(0.0),
            Delay(count) => {
                let aux = aux.unwrap();
                aux.push_back(inputs.iter().sum::<f32>());
                if aux.len() > *count as usize {
                    aux.pop_front().unwrap_or(0.0)
                } else {
                    0.0
                }
            }
            AbsDiff => inputs
                .first()
                .map(|f| (f - inputs.iter().skip(1).sum::<f32>()).abs())
                .unwrap_or(0.0),
        }
    }
}

/// Order sensors first, inner nodes second, muscles last.
fn order_neurons(a: &Neuron, ai: usize, b: &Neuron, bi: usize) -> Ordering {
    use Neuron::*;
    match (a, b) {
        (&Sensor, &Sensor) => ai.cmp(&bi),
        (&Sensor, _) => Ordering::Less,
        (_, &Sensor) => Ordering::Greater,
        (&Muscle, &Muscle) => bi.cmp(&ai),
        (&Muscle, _) => Ordering::Greater,
        (_, &Muscle) => Ordering::Less,
        (_, _) => Ordering::Equal,
    }
}

impl Neuron {
    fn aux_storage(&self) -> Option<usize> {
        use Neuron::*;
        match self {
            Delay(x) => Some(*x as usize),
            _ => None,
        }
    }

    fn storage(&self) -> u8 {
        use Neuron::*;
        match self {
            Delay(x) => *x,
            _ => 1,
        }
    }
}

#[derive(Component, Debug)]
pub struct BitBrain {
    neurons: Vec<Neuron>,
    /// This "bit code" follows a simple format:
    ///
    /// ```ignore
    /// <input count>, <input 1>, <input 2>, ..., <input count>.
    /// ```
    ///
    /// Neurons and code are read only.
    ///
    /// Storage a and b are double-buffered storage.
    code: Vec<u8>,
    eval_count: usize,
    storage_a: Vec<f32>,
    storage_b: Vec<f32>,
    aux: Vec<VecDeque<f32>>,
}

impl BitBrain {
    pub fn new(graph: &DiGraph<Neuron, ()>) -> Option<BitBrain> {
        let count: usize = graph.node_count();
        let mut update = {
            let mut u = None;
            let mut g = graph.clone();
            for _ in 0..5 {
                match toposort(&g, None) {
                    Ok(list) => {
                        u = Some(list);
                        break;
                    }
                    Err(cycle) => {
                        let edges: Vec<_> = g
                            .edges_connecting(cycle.node_id(), cycle.neighbor_id())
                            .map(|e| e.id())
                            .collect();
                        if edges.is_empty() {
                            break;
                        }
                        for edge in edges {
                            g.remove_edge(edge);
                        }
                    }
                }
            }
            u?
        };
        update.sort_by(|ai, bi| order_neurons(&graph[*ai], ai.index(), &graph[*bi], bi.index()));

        let mut neurons: Vec<Neuron> = vec![];
        let mut code = vec![];
        let mut aux = vec![];
        for node_index in &update {
            use petgraph::Direction::*;
            let node = graph[*node_index];
            if let Some(aux_size) = node.aux_storage() {
                aux.push(VecDeque::with_capacity(aux_size + 1));
            }

            neurons.push(node);
            // This is implicit in its ordering.
            // code.push(i as u8);
            code.push(graph.edges_directed(*node_index, Incoming).count() as u8);
            for edge in graph.edges_directed(*node_index, Incoming) {
                code.push(
                    update
                        .iter()
                        .position(|n| *n == edge.source())
                        .expect("neuron position") as u8,
                );
            }
        }

        Some(BitBrain {
            neurons,
            code,
            eval_count: 0,
            storage_a: vec![0.0; count],
            storage_b: vec![0.0; count],
            aux,
        })
    }

    pub fn read_muscle(&self, index: usize) -> Option<&f32> {
        self.read()
            .get(self.storage_a.len().saturating_sub(index + 1))
    }

    /// Return the read storage.
    pub fn read(&self) -> &[f32] {
        if self.eval_count % 2 == 0 {
            &self.storage_a
        } else {
            &self.storage_b
        }
    }

    /// Return the write storage.
    pub fn write(&mut self) -> &mut [f32] {
        if self.eval_count % 2 == 1 {
            &mut self.storage_a
        } else {
            &mut self.storage_b
        }
    }

    /// Evaluate all nodes in the network in topological order.
    pub fn eval(&mut self, ctx: &Context) {
        let mut i: usize = 0;
        let mut scratch = vec![];
        let mut j = 0;
        let mut m = 0;
        while i < self.code.len() {
            // let j = self.code[i] as usize;
            let neuron = self.neurons[j];
            let count = self.code[i] as usize;
            i += 1;
            scratch.clear();
            for _ in 0..count {
                let k = self.code[i] as usize;
                scratch.push(self.read()[k]);
                i += 1;
            }
            self.write()[j] = neuron.eval(
                ctx,
                self.read()[j],
                &scratch,
                neuron.aux_storage().map(|_| {
                    let n = m;
                    m += 1;
                    &mut self.aux[n]
                }),
            );
            j += 1;
        }
        self.eval_count += 1;
    }
}

pub fn brain_mutator<R>(graph: &mut DiGraph<NVec4, ()>, rng: &mut R) -> u32
where
    R: Rng,
{
    let m = FnMutator::from(NVec4::mutate_one).with_prob(0.1);
    let nodes = mutate_all_nodes(m);
    let edges = add_edge(|_r: &mut R| ());
    let rm_edge = FnMutator::from(remove_edge);
    let weighted = WeightedMutator::new(vec![&nodes, &edges, &rm_edge
    ], &[1.0, 1.0, 1.0]);
    weighted.mutate(graph, rng)
    // nodes.mutate(graph, rng)
}

pub fn generate_brain<R>(
    sensor_count: usize,
    muscle_count: usize,
    rng: &mut R,
) -> DiGraph<NVec4, ()>
where
    R: Rng,
{
    let mut g = DiGraph::new();
    for i in 0..sensor_count {
        g.add_node(Neuron::Sensor.into());
    }
    for i in 0..muscle_count {
        g.add_node(Neuron::Muscle.into());
    }
    let m = NVec4::gen;
    let n = add_connecting_node(m, move |_r: &mut R| ()).repeat(5);
    n.mutate(&mut g, rng);
    g
}

pub fn brain_generator<R>(rng: &mut R) -> DiGraph<NVec4, ()>
where
    R: Rng,
{
    generate_brain(0, 0, rng)
}

#[cfg(test)]
mod test {
    use super::*;
    use Neuron::*;

    #[test]
    fn compact_node_indices() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let b = g.add_node(1);
        let c = g.add_node(2);
        let d = g.add_node(3);
        let e = g.add_node(4);
        assert_eq!(a.index(), 0);
        assert_eq!(b.index(), 1);
        assert_eq!(c.index(), 2);
        assert_eq!(d.index(), 3);
        assert_eq!(e.index(), 4);
    }

    #[test]
    fn cross_graph_indices() {
        let mut g = Graph::<isize, ()>::new();
        let a = g.add_node(0);
        let b = g.add_node(1);
        let h = g.clone();
        assert_eq!(h[a], 0);
        assert_eq!(h[b], 1);
    }

    #[test]
    fn topo_sort() {
        let graph = crate::brain::lessin::fig4_3();
        let brain = BitBrain::new(&graph).unwrap();
        assert_eq!(
            brain.neurons,
            vec![
                Sensor,
                Sensor,
                Sensor,
                Sin {
                    amp: 1.0,
                    freq: 1.97,
                    phase: 0.83
                },
                Complement,
                Mult,
                Mult,
                Muscle,
                Muscle,
                Muscle
            ]
        );
    }

    #[test]
    fn neuron_eval() {
        let ctx = Context { time: 0.0 };
        let inputs = [2.0, 1.0];
        assert_eq!(Sensor.eval(&ctx, 1.0, &inputs, None), 1.0);
        assert_eq!(Sum.eval(&ctx, 1.0, &inputs, None), 3.0);
        assert_eq!(Diff.eval(&ctx, 1.0, &inputs, None), 1.0);
    }

    #[test]
    fn delay_eval() {
        let ctx = Context { time: 0.0 };
        let delay = Delay(2);
        let mut aux = VecDeque::new();
        assert_eq!(aux.len(), 0);
        assert_eq!(delay.eval(&ctx, 0.0, &[1.0], Some(&mut aux)), 0.0);
        assert_eq!(aux.len(), 1);
        assert_eq!(delay.eval(&ctx, 0.0, &[0.5], Some(&mut aux)), 0.0);
        assert_eq!(aux.len(), 2);
        assert_eq!(delay.eval(&ctx, 0.0, &[0.0], Some(&mut aux)), 1.0);
        assert_eq!(aux.len(), 2);
        assert_eq!(delay.eval(&ctx, 0.0, &[0.0], Some(&mut aux)), 0.5);
        assert_eq!(aux.len(), 2);
        assert_eq!(delay.eval(&ctx, 0.0, &[0.0], Some(&mut aux)), 0.0);
        assert_eq!(aux.len(), 2);
    }

    #[test]
    fn bitbrain_eval() {
        let ctx = Context { time: 0.0 };
        let mut g = Graph::<Neuron, ()>::new();
        let _a = g.add_node(Const(1.0));
        let mut brain = BitBrain::new(&g).unwrap();
        assert_eq!(brain.storage_a.len(), 1);
        assert_eq!(brain.read()[0], 0.0);
        brain.eval(&ctx);
        assert_eq!(brain.read()[0], 1.0);
    }

    #[test]
    fn bitbrain_eval_cycle() {
        let ctx = Context { time: 0.0 };
        let mut g = Graph::<Neuron, ()>::new();
        let a = g.add_node(Const(1.0));
        let _e = g.add_edge(a, a, ());
        let mut brain = BitBrain::new(&g).unwrap();
        assert_eq!(brain.storage_a.len(), 1);
        assert_eq!(brain.read()[0], 0.0);
        brain.eval(&ctx);
        assert_eq!(brain.read()[0], 1.0);
    }

    #[test]
    fn bitbrain_eval_scc() {
        let ctx = Context { time: 0.0 };
        let mut g = Graph::<Neuron, ()>::new();
        let a = g.add_node(Const(1.0));
        let b = g.add_node(Sum);
        let _ = g.add_edge(a, b, ());
        let _ = g.add_edge(b, a, ());
        let _ = g.add_edge(b, b, ());
        let mut brain = BitBrain::new(&g).unwrap();
        assert_eq!(brain.code, [1, 1, 2, 1, 0]);
        assert_eq!(brain.storage_a, [0.0, 0.0]);
        assert_eq!(brain.storage_b, [0.0, 0.0]);
        assert_eq!(brain.storage_a.len(), 2);
        assert_eq!(brain.read()[0], 0.0);
        brain.eval(&ctx);
        assert_eq!(brain.read()[0], 1.0);
        assert_eq!(brain.storage_a, [0.0, 0.0]);
        assert_eq!(brain.storage_b, [1.0, 0.0]);
        brain.eval(&ctx);
        assert_eq!(brain.read()[0], 1.0);
        assert_eq!(brain.storage_a, [1.0, 1.0]);
        assert_eq!(brain.storage_b, [1.0, 0.0]);
        brain.eval(&ctx);
        assert_eq!(brain.read()[0], 1.0);
        assert_eq!(brain.read()[1], 2.0);
        assert_eq!(brain.storage_a, [1.0, 1.0]);
        assert_eq!(brain.storage_b, [1.0, 2.0]);
    }

    #[test]
    fn to_nvec4() {
        let n = Neuron::Sensor;
        let v: Vec4 = n.into();
        let n2: Neuron = v.into();
        assert_eq!(n, n2);
    }

    #[test]
    fn to_nvec4_altered() {
        let n = Neuron::Sensor;
        let mut v: Vec4 = n.into();
        v.x += 0.5;
        let n2: Neuron = v.into();
        assert_ne!(n, n2);
    }
}
