"""This module provides testing methods for the RL model,

which allow user to test environment integrity step by step.
These methods gives opportunities to make actions as the Agent and analyze
changes in environment, and also visualize them on a plot.

"""

from agent import Agent
from environment import HouseEnergyEnvironment
import matplotlib.pyplot as plt


class ManualTestTerminal:
    """This class builds menu in terminal,

    which contains every each action from environment and also gives
    opportunity to save logs into file, and create plot with generated data.

    """

    def __init__(self):

        self.env = HouseEnergyEnvironment()
        self.agent = Agent(env=self.env)
        self.actions = self.env.get_actions()

    def manual_testing(self):
        """Run manual testing menu to check project integrity

        managing chosen actions by yourself. Its allows user to check
        system correct behaviour through making logs into console/file.

        """

        curr_render = last_render = self.env.render()

        # create len(curr_render[0]) lists for plots
        values_for_plt = [[] for y in range(len(curr_render[0]))]

        step = 0
        file_auto_log = False
        log_file = open("Manual_Tests_v1.log", "a")
        while True:

            # ---------- build menu ----------
            sub_menu_actions = \
                '|     Available actions menu    |          Others           |\n' \
                '|-------------------------------+---------------------------|\n'

            # dynamic build depends on actions count
            i = 1
            j = len(self.actions) + 1
            for action in self.actions:
                if i == 1:
                    sub_menu_actions += \
                        '| {0:2}) {1:25} | {2:2}) File auto log: {3:5}  |\n' \
                            .format(i, action, j, str(file_auto_log))
                    j += 1
                elif i == 2:
                    sub_menu_actions += \
                        '| {0:2}) {1:25} | {2:2}) Show plots {3:10} |\n' \
                            .format(i, action, j, ' ')
                    j += 1
                elif i == 3:
                    sub_menu_actions += \
                        '| {0:2}) {1:25} | {2:2}) Nop act. for time     |\n' \
                            .format(i, action, j)
                    j += 1
                elif i == 4:
                    sub_menu_actions += \
                        '| {0:2}) {1:25} | {2:2}) Load agents model     |\n' \
                            .format(i, action, j)
                    j += 1
                elif i == 5:
                    sub_menu_actions += \
                        '| {0:2}) {1:25} | {2:2}) Let agent decide      |\n' \
                            .format(i, action, j)
                    j += 1
                elif i == 6:
                    sub_menu_actions += \
                        '| {0:2}) {1:25} | {2:2}) Exit tests {3:10} |\n' \
                            .format(i, action, j, ' ')
                elif i == 7:
                    sub_menu_actions += \
                        '| {0:2}) {1:25} |---------------------------|\n' \
                            .format(i, action)
                elif i == 8:
                    sub_menu_actions += \
                        '| {0:2}) {1:25} | Current step: {2:10}  |\n' \
                            .format(i, action, step)
                elif i == 9:
                    sub_menu_actions += \
                        '| {0:2}) {1:25} | Current time: {2:10}  |\n' \
                            .format(i, action, ' ')
                elif i == 10:
                    sub_menu_actions += \
                        '| {0:2}) {1:25} | {2}       |\n' \
                            .format(i, action, self.env.world.current_date)
                else:
                    sub_menu_actions += '| {0:2}) {1:25} | {2:25} |\n' \
                        .format(i, action, ' ')
                i += 1
            sub_menu_actions += \
                '+-------------------------------+---------------------------+\n'

            # add main menu tag
            menu = \
                '+-----------------------------------------------------------+\n' \
                '|                       Testing menu                        |\n' \
                '|-----------------------------------------------------------|\n' \
                '{0}'.format(sub_menu_actions)

            render_menu = 'Rendered values:\n'
            render_menu += \
                '+---------------------------+------------+---+------------+\n'
            render_menu += '| {0:25} |  {1:9} | {2} |  {3:9} |\n'. \
                format('Value:',
                       'Previous:',
                       '?',
                       'Current:')
            render_menu += \
                '+---------------------------+------------+---+------------+\n'
            for i in range(len(last_render[0])):
                if not isinstance(last_render[1][i], str):
                    if float(last_render[1][i]) < float(curr_render[1][i]):
                        mark = '<'
                    elif float(last_render[1][i]) > float(curr_render[1][i]):
                        mark = '>'
                    else:
                        mark = '='

                    render_menu += '| {0:25} | {1:10.4f} | {2} | {3:10.4f} |\n'. \
                        format(last_render[0][i], last_render[1][i], mark,
                               curr_render[1][i])
                else:
                    mark = '?'
                    render_menu += '| {0:25} |    {1:7} | {2} |    {3:7} |\n'. \
                        format(last_render[0][i], last_render[1][i], mark,
                               curr_render[1][i])

            render_menu += \
                '+---------------------------+------------+---+------------+\n'

            menu += render_menu

            # print build menu
            print(menu)

            # update lists for plots
            for i in range(len(curr_render[2])):
                values_for_plt[i].append(curr_render[2][i])

            if file_auto_log:
                log_file.write(render_menu)

            # ---------- build menu end ----------

            try:
                option = input('\nSelect option:\n')

                if int(option) in range(1, len(self.actions) + 1):
                    last_render = curr_render

                    if file_auto_log:
                        log_file.write(
                            '\nCurrent step: {0}\n'
                            'Chosen action: {1}\n'.format(
                                step + 1, self.actions[int(option) - 1]))

                    # pass the action with the step
                    self.env.step(self.actions[int(option) - 1])
                    curr_render = self.env.render()
                    step += 1

                elif int(option) == len(self.actions) + 1:
                    file_auto_log = not file_auto_log
                    if file_auto_log:
                        log_file.write('\n----- New Test ----\n\n')
                        step = 0
                        self.env.reset()
                        last_render = curr_render = self.env.render()
                        for i in values_for_plt:
                            i.clear()

                elif int(option) == len(self.actions) + 2:
                    # what to skipp on plot
                    skip_list = [int(x) for x in input(
                        'Enter indexes seperated by space '
                        'which should be skipped on plot:\n').split()]
                    for i in range(len(curr_render[0])):
                        if i not in skip_list:
                            plt.plot(values_for_plt[i], label=curr_render[0][i])
                    plt.legend()
                    plt.show()

                elif int(option) == len(self.actions) + 3:
                    time = float(input('Pass time in hour:\n'))
                    while time - self.env.world.time_step_in_minutes / 60 >= 0:

                        last_render = curr_render

                        # pass the action with the step
                        self.env.step('action_nop')

                        curr_render = self.env.render()
                        step += 1

                        # update lists for plots
                        for i in range(len(curr_render[2])):
                            values_for_plt[i].append(curr_render[2][i])

                        time -= self.env.world.time_step_in_minutes / 60

                elif int(option) == len(self.actions) + 4:
                    model_number = input('Enter model number to load\n')
                    self.agent.load_model_info(model_number)

                elif int(option) == len(self.actions) + 5:
                    last_render = curr_render

                    # let agent decide here for one action
                    action_index = \
                        self.agent.get_next_action_greedy(curr_render[2])

                    self.env.step(self.actions[action_index])

                    curr_render = self.env.render()
                    step += 1

                    print('Agent decided to do: {}'.format(
                        self.actions[action_index]))

                elif int(option) == len(self.actions) + 6:
                    break

                else:
                    raise ValueError()

            except ValueError:
                print("Invalid option!")

        # while end, close file and save logs
        log_file.close()
