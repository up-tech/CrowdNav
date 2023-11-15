""""
Code base on CrowdNav: url
As basic class for crowd nav simulation 
"""
import logging
import math
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
import torch
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from collections import namedtuple
from crowd_sim.envs.utils.action import ActionXY, ActionRot


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

        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            vx = human.vx - action.vx
            vy = human.vy - action.vy
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

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

    def reset_human_positon(self):
        for i in range(self.human_num):
            self.generate_circle_crossing_human(self.humans[i])

    def reset_robot_position(self):
        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)

    # px py gx gy vx vy theta
    def generate_circle_crossing_human(self, human):
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            # for agent in [self.robot] + self.humans:
            #     min_dist = human.radius + agent.radius
            #     if norm((px - agent.px, py - agent.py)) < min_dist or norm((px - agent.gx, py - agent.gy)) < min_dist:
            #         collide = True
            #         break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human
    
    def robot_interface(self):
        return self.robot
    
    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            agents_dist_circle = plt.Circle((0, 0), self.circle_radius, fill=False, color='b', linestyle='--')
            ax.add_artist(agents_dist_circle)

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

            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            
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
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        # if i == 0:
                        #     agent_state = state[0]
                        # else:
                        #     agent_state = state[i - 1]
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

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError
