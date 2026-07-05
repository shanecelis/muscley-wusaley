#[cfg(feature = "avian")]
use avian3d::{math::*, prelude::*};
#[cfg(all(feature = "avian", feature = "pickup"))]
use avian_pickup::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
#[cfg(feature = "rapier")]
use bevy_rapier3d::prelude::*;

use bevy::{
    app::RunFixedMainLoop, ecs::system::EntityCommands, prelude::*, time::run_fixed_main_schedule,
    window::WindowResolution,
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use genetic_ops::prelude::*;
use creatura::{body::*, brain::*, graph::*, math::*, *};
use petgraph::prelude::*;
use rand::{rng, rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{
    ffi::OsString,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
};

#[derive(Debug, Subcommand)]
enum Subcommands {
    /// Write creature to file path
    #[command(arg_required_else_help = true)]
    Write {
        /// The path to write
        #[arg(required = true, value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
        path: PathBuf,
    },
    #[command(arg_required_else_help = true)]
    Read {
        /// The path to read
        #[arg(required = true, value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
        path: PathBuf,
    },
}

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    subcommand: Option<Subcommands>,
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Creature {
    body: BodyGenotype,
    brain: DiGraph<Neuron, ()>,
}

fn main() {
    let cli = Cli::parse();
    let mut app = App::new();
    // if let Some(seed) = args.seed {
    //     app.insert_resource(Seed(seed));
    // }
    let mut creature = Creature {
        body: match cli.seed {
            Some(seed) => {
                let mut rng = StdRng::seed_from_u64(seed);
                BodyGenotype::gen(&mut rng)
            }
            None => snake_graph(3),
        },
        brain: {
            let mut g = DiGraph::new();
            let a = g.add_node(Neuron::Sin {
                amp: 1.0,
                freq: 1.0,
                phase: 0.0,
            });
            // let a = g.add_node(Neuron::Const(1.0));
            let b = g.add_node(Neuron::Muscle);
            g.add_edge(a, b, ());
            g
        },
    };
    if let Some(subcommand) = cli.subcommand {
        match subcommand {
            Subcommands::Write { path } => {
                let file = File::create(path).expect("file");
                let mut writer = BufWriter::new(file);
                serde_json::to_writer_pretty(&mut writer, &creature).expect("write");
                writer.flush().expect("flush");
                return ();
            }
            Subcommands::Read { path } => {
                let file = File::open(path).expect("file");
                let reader = BufReader::new(file);
                creature = serde_json::from_reader(reader).expect("parse");
            }
            _ => {}
        }
    }

    let blue = Color::srgb_u8(27, 174, 228);

    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            resolution: WindowResolution::new(400, 400),
            ..default()
        }),
        ..default()
    }));
    // Add plugins and startup system
    #[cfg(feature = "avian")]
    app.add_plugins((
        PhysicsDebugPlugin::default(),
        PhysicsPlugins::default(),
        #[cfg(feature = "pickup")]
        AvianPickupPlugin::default(),
        // Add interpolation
        // AvianInterpolationPlugin::default(),
    ));
    #[cfg(feature = "rapier")]
    app.add_plugins((
        RapierPhysicsPlugin::<NoUserData>::default(),
        RapierDebugRenderPlugin::default(),
    ));
    app.add_plugins(CreaturaPlugin)
        .insert_resource(ClearColor(blue))
        .add_systems(Startup, setup_env)
        .add_systems(Startup, (move || creature.clone()).pipe(construct_creature))
        .add_systems(Update, (mutate_on_space, delete_on_backspace))
        .add_plugins(PanOrbitCameraPlugin);

    #[cfg(feature = "pickup")]
    app.add_systems(
        RunFixedMainLoop,
        (handle_pickup_input).before(run_fixed_main_schedule),
    );
    // Run the app
    app.run();
}

/// Pass player input along to `avian_pickup`
#[cfg(feature = "pickup")]
fn handle_pickup_input(
    mut avian_pickup_input_writer: EventWriter<AvianPickupInput>,
    key_input: Res<ButtonInput<MouseButton>>,
    actors: Query<Entity, With<AvianPickupActor>>,
) {
    for actor in &actors {
        if key_input.just_pressed(MouseButton::Left) {
            avian_pickup_input_writer.send(AvianPickupInput {
                action: AvianPickupAction::Throw,
                actor,
            });
        }
        if key_input.just_pressed(MouseButton::Right) {
            avian_pickup_input_writer.send(AvianPickupInput {
                action: AvianPickupAction::Drop,
                actor,
            });
        }
        if key_input.pressed(MouseButton::Right) {
            avian_pickup_input_writer.send(AvianPickupInput {
                action: AvianPickupAction::Pull,
                actor,
            });
        }
    }
}

#[derive(Component)]
struct RootBody(Vec<Entity>);

#[derive(Resource)]
struct Seed(u64);

fn rng_from_seed(seed: Option<Res<Seed>>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed.0),
        None => StdRng::from_rng(&mut rng()),
    }
}

#[cfg(feature = "avian")]
fn make_static(mut commands: &mut EntityCommands) {
    commands.insert(RigidBody::Static);
}

#[cfg(feature = "rapier")]
fn make_static(mut commands: &mut EntityCommands) {
    commands.insert(RigidBody::Fixed);
}

fn construct_creature(
    In(input): In<Creature>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let pink = Color::srgb_u8(253, 53, 176);
    let density = 1.0;
    let genotype = input.body;
    let mut root_id = None;
    let mut muscles = vec![];

    let entities: Vec<Entity> = construct_phenotype(
        &genotype.graph,
        genotype.start,
        BuildState::default(),
        Vector3::Y,
        Vector3::X,
        Vector3::Y,
        &mut commands,
        |part, commands| {
            let id = cube_body(
                part,
                part.position,
                pink,
                density,
                &mut meshes,
                &mut materials,
                commands,
            );
            if root_id.is_none() {
                root_id = Some(id);
                make_static(&mut commands.entity(id));
            }
            Some(id)
        },
        |joint: &JointConfig, commands| Some(spherical_joint(joint, commands)),
        |a, b, commands| {
            Some({
                let id = build_muscle(a, b, commands);
                muscles.push(id);
                id
            })
        },
    )
    .expect("creature")
    .into_iter()
    .collect();

    // If the lambda above is move ||, then root_id will be None.
    if let Some(id) = root_id {
        commands.entity(id).insert(RootBody(entities));
    } else {
        warn!("No root id");
    }

    let brain = BitBrain::new(&input.brain).unwrap();
    let genotype: DiGraph<NVec4, ()> = input.brain.map(|_ni, n| (*n).into(), |_, _| ());

    commands.spawn((
        NervousSystem {
            muscles,
            sensors: vec![],
        },
        // KeyboardBrain,
        brain,
        Genotype(genotype),
    ));
}

fn delete_on_backspace(
    mut query: Query<&RootBody>,
    mut commands: Commands,
    input: Res<ButtonInput<KeyCode>>,
) {
    if input.just_pressed(KeyCode::Backspace) {
        if let Ok(creature) = query.single() {
            for id in &creature.0 {
                commands.entity(*id).despawn();
            }
        }
    }
}

fn mutate_on_space(
    mut query: Query<(&mut BitBrain, &mut Genotype<DiGraph<NVec4, ()>>)>,
    input: Res<ButtonInput<KeyCode>>,
) {
    if input.just_pressed(KeyCode::Space) {
        let mut rng = rng();
        let Ok((mut brain, mut genotype)) = query.single_mut() else {
            return;
        };
        let mut g = genotype.0.clone();
        let count = brain_mutator(&mut g, &mut rng);

        let h: DiGraph<Neuron, ()> = g.map(|_ni, n| (*n).into(), |_, _| ());
        genotype.0 = g;

        if let Some(new_brain) = BitBrain::new(&h) {
            *brain = new_brain;
            info!("brain {count} mutated; brain replaced.");
        } else {
            warn!("brain {count} mutated; brain NOT replaced.");
        }
    }

    if input.just_pressed(KeyCode::Enter) {
        let mut rng = rng();
        let Ok((mut brain, mut genotype)) = query.single_mut() else {
            return;
        };
        let g = brain_generator.gen(&mut rng);

        let h: DiGraph<Neuron, ()> = g.map(|_ni, n| (*n).into(), |_, _| ());
        genotype.0 = g;

        if let Some(new_brain) = BitBrain::new(&h) {
            *brain = new_brain;
            info!("brain generated; brain replaced.");
        } else {
            warn!("brain generated; brain NOT replaced.");
        }
    }
}

#[derive(Component)]
struct Genotype<T>(T);

#[cfg(feature = "avian")]
fn setup_plane(mut commands: EntityCommands) {
    commands.insert((
        CollisionLayers::new(
            [Layer::Ground],
            [Layer::Part, Layer::PartEven, Layer::PartOdd],
        ),
        RigidBody::Static,
        Collider::cuboid(10., 0.1, 10.),
    ));
}

#[cfg(feature = "rapier")]
fn setup_plane(mut commands: EntityCommands) {
    commands.insert((RigidBody::Fixed, Collider::cuboid(5., 0.05, 5.)));
}

fn setup_env(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let ground_color = Color::srgb_u8(226, 199, 184);
    // Ground
    setup_plane(commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(10., 0.1, 10.))),
        MeshMaterial3d(materials.add(ground_color)),
    )));

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 1000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(1.0, 8.0, 1.0).looking_at(Vec3::ZERO, Dir3::Y),
    ));

    // Camera
    let thing = commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));

    #[cfg(feature = "pickup")]
    thing.insert(
        // Add this to set up the camera as the entity that can pick up
        // objects.
        AvianPickupActor {
            // Increase the maximum distance a bit to show off the
            // prop changing its distance on scroll.
            interaction_distance: 15.0,
            ..default()
        },
        // InputAccumulation::default(),
    );
}

#[cfg(feature = "rapier")]
fn build_muscle(parent: &MuscleSite, child: &MuscleSite, commands: &mut Commands) -> Entity {
    // return child.id;
    let p_parent = parent.part.transform().transform_point(parent.anchor_local);
    let p_child = child.part.transform().transform_point(child.anchor_local);
    let rest_length = (p_parent - p_child).length();
    dbg!(rest_length);
    let spring_joint = SpringJointBuilder::new(-rest_length, 100.0, 1.0)
        .local_anchor1(parent.anchor_local)
        .local_anchor2(child.anchor_local)
        .contacts_enabled(false);
    commands
        .entity(child.id)
        .insert(ImpulseJoint::new(parent.id, spring_joint))
        .insert(Muscle { value: 0.5 })
        .insert(MuscleRange {
            min: 0.0,
            max: rest_length * 2.0,
        })
        .id()
}

#[cfg(feature = "avian")]
fn build_muscle(parent: &MuscleSite, child: &MuscleSite, commands: &mut Commands) -> Entity {
    let p_parent = parent.part.transform().transform_point(parent.anchor_local);
    let p_child = child.part.transform().transform_point(child.anchor_local);
    let rest_length = (p_parent - p_child).length();
    commands
        .spawn(
            DistanceJoint::new(parent.id, child.id)
                .with_local_anchor1(parent.anchor_local)
                .with_local_anchor2(child.anchor_local)
                .with_limits(dbg!(rest_length), dbg!(rest_length))
                // .with_limits(rest_length, rest_length)
                // .with_linear_velocity_damping(0.1)
                // .with_angular_velocity_damping(1.0)
                .with_compliance(1.0 / 100.0),
        )
        .insert(Muscle { value: 0.5 })
        .insert(MuscleRange {
            min: 0.0,
            max: rest_length * 2.0,
        })
        .id()
}
