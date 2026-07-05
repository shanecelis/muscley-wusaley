use crate::stamp::*;
#[cfg(feature = "avian")]
use avian3d::{math::*, prelude::*};
use bevy::prelude::*;
#[cfg(feature = "rapier")]
use bevy_rapier3d::prelude::*;
pub mod body;
pub mod brain;

// pub mod operator;
mod repeat_visit_map;
pub mod stamp;
use std::f32::consts::TAU;

pub mod graph;
pub mod math;
pub mod rdfs;

#[derive(Component)]
pub struct MuscleRange {
    pub min: math::Scalar,
    pub max: math::Scalar,
}

/// Use an even and odd part scheme so that the root part is even. Every part
/// successively attached is odd then even then odd. Then we don't allow even
/// and odd parts to collide. This is how we can create our own "no collisions
/// between objects that share a joint."
#[cfg_attr(feature = "avian", derive(PhysicsLayer))]
#[derive(Default)]
pub enum Layer {
    Ground,
    #[default]
    Part,
    PartEven,
    PartOdd,
}

/// Sensor `value` $\in [0, 1]$.
#[derive(Component, Default)]
pub struct Sensor {
    pub value: math::Scalar,
}

/// Muscle `value` $\in [0, 1]$.
#[derive(Component, Default)]
pub struct Muscle {
    pub value: math::Scalar,
}

impl Muscle {
    pub fn apply(&self, range: Option<&MuscleRange>) -> math::Scalar {
        let value = self.value.clamp(0.0, 1.0);
        range
            .map(|range| {
                let delta = range.max - range.min;
                value * delta + range.min
            })
            .unwrap_or(value)
    }
}

pub struct CreaturaPlugin;

impl Plugin for CreaturaPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(brain::plugin)
            .add_systems(Update, oscillate_muscles)
            .add_systems(Update, oscillate_brain)
            .add_systems(Update, keyboard_brain)
            .add_systems(FixedUpdate, sync_muscles);
    }
}

#[cfg(feature = "rapier")]
#[allow(clippy::type_complexity)]
pub fn sync_muscles(
    mut joints: Query<
        (
            &mut ImpulseJoint,
            &Muscle,
            Option<&MuscleRange>,
            Option<&mut Sensor>,
        ),
        Changed<Muscle>,
    >,
) {
    for (mut joint, muscle, range, sensor) in &mut joints {
        // if let Some(mut sensor) = sensor {
        //     if let Some(range) = range {
        //         let delta = range.max - range.min;
        //         sensor.value = (joint.rest_length - range.min) / delta;
        //     } else {
        //         sensor.value = joint.rest_length;
        //     }
        // }
        let rest_length = muscle.apply(range);
        match &mut joint.data {
            TypedJoint::SpringJoint(spring) => {
                spring
                    .data
                    .set_motor_position(JointAxis::LinX, -rest_length, 100.0, 1.0);
            }
            _ => {
                panic!();
            }
        }
    }
}

#[cfg(feature = "avian")]
#[allow(clippy::type_complexity)]
pub fn sync_muscles(
    mut joints: Query<
        (
            &mut DistanceJoint,
            &Muscle,
            Option<&MuscleRange>,
            Option<&mut Sensor>,
        ),
        Changed<Muscle>,
    >,
) {
    for (mut joint, muscle, range, sensor) in &mut joints {
        if let Some(mut sensor) = sensor {
            if let Some(range) = range {
                let delta = range.max - range.min;
                let rest_length = joint.limits.min;
                sensor.value = (rest_length - range.min) / delta;
            } else {
                sensor.value = joint.limits.min;
            }
        }

        let rest_length = muscle.apply(range);
        joint.limits = DistanceLimit::new(rest_length, rest_length);
    }
}

pub fn oscillate_muscles(
    time: Res<Time>,
    nervous_systems: Query<(&NervousSystem, &SpringOscillator)>,
    mut muscles: Query<&mut Muscle>,
) {
    let seconds = time.elapsed_secs();
    for (nervous_system, oscillator) in &nervous_systems {
        let v = oscillator.eval(seconds);
        for muscle_id in &nervous_system.muscles {
            if let Ok(mut muscle) = muscles.get_mut(*muscle_id) {
                // eprintln!("set muscle value {v}");
                muscle.value = v;
            }
        }
    }
}

pub fn oscillate_brain(
    time: Res<Time>,
    nervous_systems: Query<(&NervousSystem, &OscillatorBrain)>,
    mut muscles: Query<&mut Muscle>,
) {
    let seconds = time.elapsed_secs();
    for (nervous_system, brain) in &nervous_systems {
        let n = brain.oscillators.len();
        for (i, muscle_id) in nervous_system.muscles.iter().enumerate() {
            if let Ok(mut muscle) = muscles.get_mut(*muscle_id) {
                let v = brain.oscillators[i % n].eval(seconds);
                muscle.value = v;
            }
        }
    }
}

#[derive(Component)]
pub struct KeyboardBrain;

#[cfg(feature = "avian")]
pub fn keyboard_brain(
    time: Res<Time>,
    input: Res<ButtonInput<KeyCode>>,
    nervous_systems: Query<&NervousSystem, With<KeyboardBrain>>,
    mut joints: Query<&mut Muscle>,
    disabled: Query<&JointDisabled>,
    mut commands: Commands,
) {
    use KeyCode::*;
    let keys = [
        [KeyQ, KeyA, KeyZ, Digit1],
        [KeyW, KeyS, KeyX, Digit2],
        [KeyE, KeyD, KeyC, Digit3],
        [KeyR, KeyF, KeyV, Digit4],
        [KeyT, KeyG, KeyB, Digit5],
        [KeyY, KeyJ, KeyN, Digit6],
    ];
    let delta = 0.1;
    for nervous_system in &nervous_systems {
        let muscles = &nervous_system.muscles;
        for i in 0..muscles.len().min(keys.len()) {
            if let Ok(mut muscle) = joints.get_mut(muscles[i]) {
                if input.pressed(keys[i][0]) {
                    muscle.value += delta * time.delta_secs();
                    eprintln!("inc muscle value {}", muscle.value);
                } else if input.pressed(keys[i][1]) {
                    muscle.value = 0.5;
                    eprintln!("reset muscle value {}", muscle.value);
                } else if input.pressed(keys[i][2]) {
                    muscle.value -= delta * time.delta_secs();
                    eprintln!("dec muscle value {}", muscle.value);
                } else if input.just_pressed(keys[i][3]) {
                    if disabled.get(muscles[i]).is_ok() {
                        commands.entity(muscles[i]).remove::<JointDisabled>();
                    } else {
                        commands.entity(muscles[i]).insert(JointDisabled);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "rapier")]
pub fn keyboard_brain(
    time: Res<Time>,
    input: Res<ButtonInput<KeyCode>>,
    nervous_systems: Query<&NervousSystem, With<KeyboardBrain>>,
    mut joints: Query<&mut Muscle>,
    // disabled: Query<&JointDisabled>,
    mut commands: Commands,
) {
    use KeyCode::*;
    let keys = [
        [KeyQ, KeyA, KeyZ, Digit1],
        [KeyW, KeyS, KeyX, Digit2],
        [KeyE, KeyD, KeyC, Digit3],
        [KeyR, KeyF, KeyV, Digit4],
        [KeyT, KeyG, KeyB, Digit5],
        [KeyY, KeyJ, KeyN, Digit6],
    ];
    let delta = 0.1;
    for nervous_system in &nervous_systems {
        let muscles = &nervous_system.muscles;
        for i in 0..muscles.len().min(keys.len()) {
            if let Ok(mut muscle) = joints.get_mut(muscles[i]) {
                if input.pressed(keys[i][0]) {
                    muscle.value += delta * time.delta_secs();
                    eprintln!("inc muscle value {}", muscle.value);
                } else if input.pressed(keys[i][1]) {
                    muscle.value = 0.5;
                    eprintln!("reset muscle value {}", muscle.value);
                } else if input.pressed(keys[i][2]) {
                    muscle.value -= delta * time.delta_secs();
                    eprintln!("dec muscle value {}", muscle.value);
                } else if input.just_pressed(keys[i][3]) {
                    warn!("No joint disabling yet for rapier");
                    // if disabled.get(muscles[i]).is_ok() {
                    //     commands.entity(muscles[i]).remove::<JointDisabled>();
                    // } else {
                    //     commands.entity(muscles[i]).insert(JointDisabled);
                    // }
                }
            }
        }
    }
}

#[derive(Component)]
pub struct OscillatorBrain {
    pub oscillators: Vec<SpringOscillator>,
}

/// Contain sensors and muscles
#[derive(Component)]
pub struct NervousSystem {
    pub sensors: Vec<Entity>,
    pub muscles: Vec<Entity>,
}

#[derive(Component, Debug)]
pub struct SpringOscillator {
    pub freq: math::Scalar,
    pub phase: math::Scalar,
}

impl SpringOscillator {
    pub fn eval(&self, seconds: f32) -> f32 {
        (TAU * self.freq * seconds + self.phase).sin()
    }
}
