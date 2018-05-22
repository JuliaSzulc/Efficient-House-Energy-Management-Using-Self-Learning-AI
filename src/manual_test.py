"""This module provides testing methods for the RL model,

which allow user to test environment integrity step by step.
These methods gives opportunities to make actions as the Agent and analyze
changes in environment, and also visualize them on a plot.

"""

from agent import Agent
from environment import HouseEnergyEnvironment
import matplotlib.pyplot as plt
from main import load_model


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
        """Runs manual testing menu to check project integrity.
        User can choose actions and other menu options, what allows to check
        correct behaviour of environment. Runs in console.
        """

        curr_state = last_state = self.env.get_current_state()

        # create len(curr_state) lists for plots
        values_for_plt = [[] for _ in curr_state.keys()]

        step = 0
        file_auto_log = False
        log_file = open("Manual_Tests.log", "a")

        while True:

            # Print Main Menu
            print(self._draw_menu(file_auto_log, step))

            # Print State Values
            state_menu = self._draw_state(curr_state, last_state)
            print(state_menu)

            # Update lists for plots
            serialized_state = self.env.serialize_state(curr_state.copy())
            for i in range(len(serialized_state)):
                values_for_plt[i].append(serialized_state[i])

            if file_auto_log:
                log_file.write(state_menu)

            # Selecting option
            try:
                option = input('\nSelect option:\n')

                if int(option) in range(1, len(self.actions) + 1):
                    last_state = self.env.get_current_state()

                    # pass the action with the step & inc step counter
                    self.env.step(self.actions[int(option) - 1])
                    step += 1

                    if file_auto_log:
                        log_file.write(
                            '\nCurrent step: {0}\n'
                            'Chosen action: {1}\n'.format(
                                step, self.actions[int(option) - 1]))

                elif int(option) == len(self.actions) + 1:
                    file_auto_log = not file_auto_log
                    if file_auto_log:
                        log_file.write('\n----- Logging ON ----\n\n')
                    else:
                        log_file.write('\n----- Logging OFF ----\n\n')

                elif int(option) == len(self.actions) + 2:

                    skip_list = [int(x) for x in input(
                        'Enter indexes separated by space '
                        'which should be skipped on plot:\n').split()]

                    for i, key in enumerate(self.env.get_current_state()
                                            .keys()):
                        if i not in skip_list:
                            plt.plot(values_for_plt[i], label=key)
                    plt.legend()
                    plt.show()

                elif int(option) == len(self.actions) + 3:
                    time = float(input('Pass time in hour:\n'))
                    while time - self.env.world.time_step_in_minutes / 60 >= 0:

                        last_state = curr_state

                        # pass the action with the step
                        self.env.step('action_nop')

                        curr_state = self.env.get_current_state()
                        serialized_state = self.env.serialize_state(
                            curr_state.copy())
                        step += 1

                        # update lists for plots
                        for i, val in enumerate(serialized_state):
                            values_for_plt[i].append(val)

                        time -= self.env.world.time_step_in_minutes / 60

                    if file_auto_log:
                        log_file.write(
                            '\nCurrent step: {0}\n'
                            'After waiting (nop action) for: {1} hours\n'.format(
                                step, time))

                elif int(option) == len(self.actions) + 4:
                    model_id = input('Enter model number to load\n')
                    load_model(self.agent, model_id)

                    print('Model {} was successfully loaded.'.format(
                        str(model_id)))

                    if file_auto_log:
                        log_file.write('Model {} was successfully loaded.'
                                       .format(str(model_id)))

                elif int(option) == len(self.actions) + 5:
                    last_state = curr_state

                    serialized_state = self.env.serialize_state(
                        curr_state.copy())
                    # let agent decide here for one action
                    action_index = \
                        self.agent.get_next_action_greedy(serialized_state)

                    self.env.step(self.actions[action_index])

                    curr_state = self.env.get_current_state()
                    step += 1

                    print('Agent decided to do: {}'.format(
                        self.actions[action_index]))

                    if file_auto_log:
                        log_file.write(
                            '\nCurrent step: {0}\n'
                            'Agent decided to do: {1}\n'.format(
                                step, self.actions[action_index]))

                elif int(option) == len(self.actions) + 6:
                    step = 0
                    self.env.reset()
                    last_state = curr_state = self.env.get_current_state()
                    for i in values_for_plt:
                        i.clear()

                    if file_auto_log:
                        log_file.write('Reset environment.\n')

                elif int(option) == len(self.actions) + 7:
                    break

                else:
                    raise ValueError()

            except ValueError:
                print("Invalid option!")

            if file_auto_log:
                log_file.write(self._draw_state(curr_state, last_state))

        # while end, close file and save logs
        log_file.close()

    def _draw_menu(self, file_auto_log, step):

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
                    '| {0:2}) {1:25} | {2:2}) Reset Environment     |\n' \
                        .format(i, action, j)
                j += 1
            elif i == 7:
                sub_menu_actions += \
                    '| {0:2}) {1:25} | {2:2}) Exit tests {3:10} |\n' \
                        .format(i, action, j, ' ')
            elif i == 8:
                sub_menu_actions += \
                    '| {0:2}) {1:25} |---------------------------|\n' \
                        .format(i, action)
            elif i == 9:
                sub_menu_actions += \
                    '| {0:2}) {1:25} | Current step: {2:10}  |\n' \
                        .format(i, action, step)
            elif i == 10:
                sub_menu_actions += \
                    '| {0:2}) {1:25} | Current time: {2:10}  |\n' \
                        .format(i, action, ' ')
            elif i == 11:
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

        return menu

    @staticmethod
    def _draw_state(curr_state, last_state):

        state_menu = 'Rendered values:\n'
        state_menu += \
            '+---------------------------+------------+---+------------+\n'
        state_menu += '| {0:25} |  {1:9} | {2} |  {3:9} |\n'. \
            format('Value:',
                   'Previous:',
                   '?',
                   'Current:')
        state_menu += \
            '+---------------------------+------------+---+------------+\n'
        for key, value in last_state.items():
            if not isinstance(value, str):
                if float(value) < float(curr_state[key]):
                    mark = '<'
                elif float(value) > float(curr_state[key]):
                    mark = '>'
                else:
                    mark = '='

                state_menu += '| {0:25} | {1:10.4f} | {2} | {3:10.4f} |\n'. \
                    format(key, value, mark,
                           curr_state[key])
            else:
                mark = '?'
                state_menu += '| {0:25} |    {1:7} | {2} |    {3:7} |\n'. \
                    format(key, value, mark,
                           curr_state[key])

        state_menu += \
            '+---------------------------+------------+---+------------+\n'

        return state_menu


def main():
    print('### MANUAL TEST MENU ###\n')
    new_test = ManualTestTerminal()
    new_test.manual_testing()


if __name__ == "__main__":
    main()
