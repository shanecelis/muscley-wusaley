use super::*;
use crate::{body::*, math::*, rdfs::*};
use bevy::ecs::system::EntityCommands;
#[cfg(feature = "rapier")]
use bevy_rapier3d::prelude::*;
use core::f32::consts::{FRAC_PI_4, PI};
use petgraph::{graph::DefaultIx, prelude::*};

#[derive(Clone, Copy, Debug)]
pub enum ConstructError {}

#[derive(Clone, Copy, Debug)]
pub struct BuildState {
    pub scale: Vector3,
    pub rotation: Quaternion,
}

impl Default for BuildState {
    fn default() -> Self {
        BuildState {
            scale: Vector3::ONE,
            rotation: Quaternion::IDENTITY,
        }
    }
}

#[derive(Debug)]
pub struct MuscleSite<'a> {
    pub id: Entity,
    pub anchor_local: Vector3,
    pub part: &'a Part,
}

#[allow(clippy::too_many_arguments)]
pub fn construct_phenotype<F, G, H>(
    graph: &DiGraph<Part, PartEdge>,
    root: NodeIndex<DefaultIx>,
    state: BuildState,
    position: Vector3,
    principle_axis: Vector3,
    secondary_axis: Vector3,
    commands: &mut Commands,
    mut make_part: F,
    mut make_joint: G,
    mut make_muscle: H,
) -> Result<Vec<Entity>, ConstructError>
where
    F: FnMut(&Part, &mut Commands) -> Option<Entity>,
    G: FnMut(&JointConfig, &mut Commands) -> Option<Entity>,
    H: FnMut(&MuscleSite, &MuscleSite, &mut Commands) -> Option<Entity>,
{
    let mut rdfs = Rdfs::new(graph, root, |g, _n, e| {
        Permit::EdgeCount(g[e].iteration_count)
    });
    let mut states = vec![];
    let mut parts: Vec<(Part, Entity)> = vec![];
    let mut entities = vec![];
    while let Some(node) = rdfs.next(graph) {
        let depth = rdfs.depth();
        let _ = states.drain(depth..);
        let _ = parts.drain(depth..);
        let mut state: BuildState = *states.last().unwrap_or(&state);
        let mut joint_rotation = None;
        if let Some(edge) = rdfs.path.last() {
            let part_edge = &graph[*edge];
            state.rotation *= part_edge.rotation;
            state.scale *= part_edge.scale;
            joint_rotation = Some(part_edge.joint_rotation);
        }
        let mut child: Part = graph[node];
        let child_id = if let Some((parent, parent_id)) = parts.last() {
            // Child object, has a parent.
            let joint_rotation = joint_rotation
                .map(|q| q * state.rotation)
                .unwrap_or(state.rotation);
            let joint_dir = Dir3::new(joint_rotation * principle_axis).expect("joint_dir");
            // child.position = parent.position + 10.0 * joint_dir;
            child.extents = state.scale * graph[node].extents;
            // Position child
            if let Some(stamp_info) = child.stamp(parent, joint_dir) {
                child.position = parent.position + stamp_info.stamp_delta;
                let child_id = make_part(&child, commands).unwrap();
                if let Some(e) = make_joint(
                    &JointConfig {
                        parent: *parent_id,
                        parent_anchor: stamp_info.surface_anchor,
                        child: child_id,
                        child_anchor: stamp_info.stamp_anchor,
                        normal: *joint_dir,
                        tangent: joint_rotation * secondary_axis,
                    },
                    commands,
                ) {
                    entities.push(e);
                }
                // Make muscles
                if let Some(edge) = rdfs.path.last() {
                    for muscle in &graph[*edge].muscles {
                        let r = joint_rotation * muscle.parent;
                        let parent_anchor_dir: Dir3 = Dir3::new(r * principle_axis).expect("dir3");
                        let child_anchor_dir: Dir3 =
                            Dir3::new(r * muscle.child * principle_axis).expect("dir3");
                        // dbg!(parent);
                        // dbg!(parent_anchor_dir);
                        // dbg!(child);
                        // dbg!(child_anchor_dir);
                        if let Some((a1, a2)) = parent
                            .cast_to(parent_anchor_dir)
                            .zip(child.cast_to(child_anchor_dir))
                        {
                            let parent_site = MuscleSite {
                                id: *parent_id,
                                anchor_local: a1,
                                part: parent,
                            };

                            let child_site = MuscleSite {
                                id: child_id,
                                anchor_local: a2,
                                part: &child,
                            };
                            // dbg!(&parent_site);
                            // dbg!(&child_site);
                            if let Some(e) = make_muscle(&parent_site, &child_site, commands) {
                                entities.push(e);
                            }
                        }
                    }
                }
                child_id
            } else {
                panic!();
            }
        } else {
            // Root object, no parent.
            child.position = position;
            make_part(&child, commands).unwrap()
        };

        #[cfg(feature = "avian")]
        commands.entity(child_id).insert(if depth % 2 == 0 {
            CollisionLayers::new([Layer::PartEven], [Layer::Ground, Layer::PartEven])
        } else {
            CollisionLayers::new([Layer::PartOdd], [Layer::Ground, Layer::PartOdd])
        });
        entities.push(child_id);
        parts.push((child, child_id));
        states.push(state);
    }
    Ok(entities)
}

pub struct JointConfig {
    pub parent: Entity,
    /// Local coordinates
    pub parent_anchor: Vector3,
    pub child: Entity,
    /// Local coordinates
    pub child_anchor: Vector3,
    /// Normal axis
    pub normal: Vector3,
    /// Tangent axis
    pub tangent: Vector3,
}

#[cfg(feature = "avian")]
pub fn spherical_joint(joint: &JointConfig, commands: &mut Commands) -> Entity {
    commands
        .spawn({
            let mut j = SphericalJoint::new(joint.parent, joint.child)
                .with_local_anchor1(joint.parent_anchor)
                .with_local_anchor2(joint.child_anchor)
                .with_swing_limits(-FRAC_PI_4, FRAC_PI_4)
                .with_twist_limits(-FRAC_PI_4, FRAC_PI_4);
            j.twist_axis = joint.normal;
            j
        })
        .id()
}

// #[cfg(feature = "rapier")]
// pub fn spherical_joint(joint: &JointConfig, commands: &mut Commands) -> Entity {
//     let spherical_joint = SphericalJointBuilder::new()
//         .local_anchor2(joint.parent_anchor)
//         .local_anchor1(joint.child_anchor)
//         .limits(JointAxis::AngY, [-FRAC_PI_4, FRAC_PI_4])
//         .limits(JointAxis::AngZ, [-FRAC_PI_4, FRAC_PI_4])
//         .limits(JointAxis::AngX, [0.0, 0.0])
//         ;
//     commands
//         .entity(joint.child)
//         .insert(ImpulseJoint::new(joint.parent, spherical_joint))
//         .id()
// }
#[cfg(feature = "rapier")]
pub fn spherical_joint(joint: &JointConfig, commands: &mut Commands) -> Entity {
    let mut spherical_joint = SphericalJointBuilder::new()
        .local_anchor1(joint.parent_anchor)
        .local_anchor2(joint.child_anchor)
        .limits(JointAxis::AngZ, [-FRAC_PI_4, FRAC_PI_4])
        .limits(JointAxis::AngX, [-FRAC_PI_4, FRAC_PI_4])
        .limits(JointAxis::AngY, [0.0, 0.0])
        .build()
        // .contacts_enabled(false)
        ;
    spherical_joint.data.set_contacts_enabled(false);
    commands
        .entity(joint.child)
        .insert(MultibodyJoint::new(joint.parent, spherical_joint.into()))
        .id()
}

#[cfg(feature = "rapier")]
pub fn revolute_joint(joint: &JointConfig, commands: &mut Commands) -> Entity {
    // let spherical_joint = RevoluteJointBuilder::new(joint.tangent)
    let mut spherical_joint = RevoluteJointBuilder::new(Vec3::Z)
        .local_anchor1(joint.parent_anchor)
        .local_anchor2(joint.child_anchor)
        .limits([-FRAC_PI_4, FRAC_PI_4])
        .build()
        // .contacts_enabled(false)
        ;
    spherical_joint.data.set_contacts_enabled(false);
    commands
        .entity(joint.child)
        .insert(MultibodyJoint::new(joint.parent, spherical_joint.into()))
        .id()
}

#[cfg(feature = "avian")]
fn make_dynamic(mut commands: &mut EntityCommands, child: &Part, position: Vec3, density: f32) {
    commands.insert((
        // RigidBody::Static,
        RigidBody::Dynamic,
        Position(position),
        MassPropertiesBundle::from_shape(&child.collider(), child.volume() * density),
        // c,
        child.collider(),
    ));
}

#[cfg(feature = "rapier")]
fn make_dynamic(mut commands: &mut EntityCommands, child: &Part, position: Vec3, density: f32) {
    commands.insert((
        RigidBody::Dynamic,
        TransformBundle::from(Transform::from_translation(position)),
        ColliderMassProperties::Density(density),
        child.collider(),
    ));
}

pub fn cube_body(
    child: &Part,
    position: Vec3,
    color: Color,
    density: f32,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    commands: &mut Commands,
) -> Entity {
    let mut entity_commands = commands.spawn((
        Mesh3d(meshes.add(Mesh::from(child.shape()))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: color,
            ..default()
        })),
    ));
    make_dynamic(&mut entity_commands, child, position, density);
    entity_commands.id()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cast_to() {
        let parent = Part {
            extents: Vec3::ONE,
            position: Vec3::Y,
            rotation: Quat::IDENTITY,
        };

        let anchor_dir = Dir3::Y;
        let hit = parent.cast_to(anchor_dir).unwrap();
        assert_eq!(hit, Vec3::Y / 2.0);
    }

    #[test]
    fn quat_div() {
        let a = Quat::from_axis_angle(Vec3::X, PI);
        let b = Quat::from_axis_angle(Vec3::X, PI / 2.0);
        let c = a * b.inverse();
        assert!(c.angle_between(b) < 0.01);
    }
}
