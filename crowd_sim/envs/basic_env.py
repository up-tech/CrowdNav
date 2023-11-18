""""
Code base on CrowdNav: url
As basic class for crowd nav simulation 
"""

import gym
import matplotlib.lines as mlines
import numpy as np
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *
import matplotlib.pyplot as plt
from matplotlib import animation

class BasicEnv(gym.Env):

    def __init__(self):
        super(BasicEnv, self).__init__()
        self.time_limit = None
        self.time_step = None
        self.robot = None  # agent robot
        self.humans = None # agent humans
        self.global_time = None

        # simulation configuration
        self.config = None
        self.circle_radius = None
        self.human_num = None

        # for visualization
        self.states = None
        self.action_values = None

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.social_dist = config.getfloat('env', 'social_dist')

        self.circle_radius = config.getfloat('sim', 'circle_radius')
        self.human_num = config.getint('sim', 'human_num')

        self.generate_agents(self.config)

    def reset(self, seed = None):

        np.random.seed(seed)

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        if self.humans is None:
            raise AttributeError('humans has to be set!')
     
        self.global_time = 0

        self.reset_robot_position()
        self.reset_human_positon()

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()

        # get initial observation
        ob = [self.robot.get_full_state()] + [human.get_full_state() for human in self.humans]

        self.states.append(ob)
        return ob
    
    def step(self, action):
        done = False

        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_full_state() for other_human in self.humans if other_human != human]
            
            if self.robot.visible:
                ob += [self.robot.get_full_state()]

            human_actions.append(human.act(ob))

        # for i, human in enumerate(self.humans):
        #     px = human.px - self.robot.px
        #     py = human.py - self.robot.py
        #     vx = human.vx - action.vx
        #     vy = human.vy - action.vy

        # update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        
        self.global_time += self.time_step

        if self.global_time > self.time_limit:
            done = True
        if self.robot.reached_destination():
            done = True

        ob = [self.robot.get_full_state()] + [human.get_full_state() for human in self.humans]
        
        self.states.append(ob)

        reward = 0
        info = {}

        return ob, reward, done, info

    def generate_agents(self, config):
        robot = Robot(config, 'robot')
        #robot.set_policy()
        self.robot = robot
        self.humans = []
        for i in range(self.human_num):
            human = Human(self.config, 'humans')
            #human.set_policy()
            self.humans.append(human)

    def reset_robot_position(self):
        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)

    def reset_human_positon(self):
        for human in self.humans:
            human.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        for human in self.humans:
            self.generate_circle_crossing_human(human)

    def generate_circle_crossing_human(self, human):
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.social_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human
    
    def robot_interface(self):
        return self.robot
    
    def render(self, mode = None):

        #human num offset
        x_offset = 0.11
        y_offset = 0.11
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.tick_params(labelsize=16)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)

        agents_distribution_circle = plt.Circle((0, 0), self.circle_radius, fill=False, color='b', linestyle='--')
        ax.add_artist(agents_distribution_circle)

        # add robot and its goal
        robot_positions = [state[0].position for state in self.states]
        goal = mlines.Line2D([0], [self.circle_radius], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)

        ax.add_artist(robot)
        ax.add_artist(goal)
        # add a legend in the upper right corner
        plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

        # add humans and their numbers
        human_positions = [[state[j].position for j in range(1, len(self.humans) + 1)] for state in self.states]

        humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False) for i in range(len(self.humans))]
        
        human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                    color='black', fontsize=12) for i in range(len(self.humans))]
        for i, human in enumerate(humans):
            ax.add_artist(human)
            ax.add_artist(human_numbers[i])

        # add time annotation
        time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
        ax.add_artist(time)

        # compute orientation in each step and use arrow to show the direction
        radius = self.robot.radius
        orientations = []
        for i in range(self.human_num + 1):
            orientation = []
            for state in self.states:
                agent_state = state[i]
                theta = np.arctan2(agent_state.vy, agent_state.vx)
                orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                        agent_state.py + radius * np.sin(theta))))
            orientations.append(orientation)
        arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                    for orientation in orientations]
        for arrow in arrows:
            ax.add_artist(arrow)
        global_step = 0

        def update(frame_num):
            nonlocal global_step
            nonlocal arrows
            global_step = frame_num
            robot.center = robot_positions[frame_num]
            for i, human in enumerate(humans):
                human.center = human_positions[frame_num][i]
                human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                for arrow in arrows:
                    arrow.remove()
                arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                    arrowstyle=arrow_style) for orientation in orientations]
                for arrow in arrows:
                    ax.add_artist(arrow)

            time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

        anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)

        anim.running = True
        
        plt.show()
