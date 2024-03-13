from time import sleep

from klampt import vis, WorldModel
from klampt.model import coordinates, trajectory
from jrl.robot import Robot
from klampt.model.collide import WorldCollider


def set_environment(robot: Robot, obstacles):
    world = robot.klampt_world_model
    world.loadTerrain(r"data/terrains/block.off")
    robot._ignored_collision_pairs_formatted += [
        (robot._klampt_robot.link(robot._base_link), world.terrain(i))
        for i in range(world.numTerrains())]
    for obs in obstacles:
        world.loadRigidObject(obs)
    if not robot.klampt_world_model.numRigidObjects() == len(robot._klampt_collision_checker.rigidObjects):
        print(robot._ignored_collision_pairs_formatted)
        robot._klampt_collision_checker = WorldCollider(
            robot._klampt_world_model,
            ignore=robot._ignored_collision_pairs_formatted
        )
        print("reinitialize the world collider, because of the unmatch between world and collider",
              robot.klampt_world_model.numRigidObjects(),
              len(robot._klampt_collision_checker.rigidObjects))
    return world


def show_world(world: WorldModel, window_title):
    vis.init()
    vis.add("world", world)
    vis.add("coordinates", coordinates.manager())
    vis.setWindowTitle(window_title)
    vis.show()
    while vis.shown():
        vis.lock()
        vis.unlock()
        sleep(1 / 30)  # note: don't put sleep inside the lock()
    vis.kill()


def _init_klampt_vis(robot: Robot, window_title: str, show_collision_capsules: bool = True):
    vis.init()

    background_color = (1, 1, 1, 0.7)
    vis.setBackgroundColor(background_color[0], background_color[1], background_color[2], background_color[3])
    size = 5
    for x0 in range(-size, size + 1):
        for y0 in range(-size, size + 1):
            vis.add(
                f"floor_{x0}_{y0}",
                trajectory.Trajectory([1, 0], [(-size, y0, 0), (size, y0, 0)]),
                color=(0.75, 0.75, 0.75, 1.0),
                width=2.0,
                hide_label=True,
                pointSize=0,
            )
            vis.add(
                f"floor_{x0}_{y0}2",
                trajectory.Trajectory([1, 0], [(x0, -size, 0), (x0, size, 0)]),
                color=(0.75, 0.75, 0.75, 1.0),
                width=2.0,
                hide_label=True,
                pointSize=0,
            )

    vis.add("world", robot.klampt_world_model)
    vis.add("coordinates", coordinates.manager())
    vis.setWindowTitle(window_title)
    vis.show()

