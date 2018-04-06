"""This it the main executable module for the project.

it sets RL-agent along with environment, prints the resultes to user and
database etc.

"""
from agent import Agent
from environment import HouseEnergyEnvironment
import matplotlib.pyplot as plt


def save_to_database(info):
    """Save all the information about an experiment to db"""
    print(info)


def manual_testing():
    """Run manual testing menu to check project integrity

    managing chosen actions by yourself. Its allows user to check
    system correct behaviour through making logs into console/file.

    Args:
        env (HouseEnergyEnvironment) - object to operate
        actions (list) - available actions to make
        file_auto_log (bool) - true/false, says if logs should be saved to file
        stage (number) - count of passed steps in env
        last_render (tuple) - names ans values of env parameters
        log_file (file) - output for logs

    """

    env = HouseEnergyEnvironment()
    actions = env.get_actions()
    curr_render = last_render = env.render()

    # create len(curr_render[0]) lists for plots
    values_for_plt = [[] for y in range(len(curr_render[0]))]

    step = 0
    file_auto_log = False
    log_file = open("Manual_Tests_v1.log", "a")
    while True:

        # ---------- build menu ----------
        sub_menu_actions = \
            '|     Available actions menu    |          Others          |\n' \
            '|-------------------------------+--------------------------|\n'

        # dynamic build depends on actions count
        i = 1
        j = len(actions) + 1
        for action in actions:
            if i == 1:
                sub_menu_actions += \
                    '| {0:2}) {1:25} | {2:2}) File auto log: {3:5} |\n' \
                        .format(i, action, j, str(file_auto_log))
                j += 1
            elif i == 2:
                sub_menu_actions += \
                    '| {0:2}) {1:25} | {2:2}) Show plots {3:10}|\n' \
                        .format(i, action, j, ' ')
                j += 1
            elif i == 3:
                sub_menu_actions += \
                    '| {0:2}) {1:25} | {2:2}) Nop act. for time    |\n' \
                        .format(i, action, j)
                j += 1
            elif i == 4:
                sub_menu_actions += \
                    '| {0:2}) {1:25} | {2:2}) Exit tests {3:10}|\n' \
                        .format(i, action, j, ' ')
                j += 1
            elif i == 5:
                sub_menu_actions += \
                    '| {0:2}) {1:25} |--------------------------|\n' \
                        .format(i, action)
            elif i == 6:
                sub_menu_actions += \
                    '| {0:2}) {1:25} | Current step: {2:10} |\n' \
                        .format(i, action, step)
            elif i == 7:
                sub_menu_actions += \
                    '| {0:2}) {1:25} | Current time: {2:10} |\n' \
                        .format(i, action, ' ')
            elif i == 8:
                sub_menu_actions += \
                    '| {0:2}) {1:25} | {2}      |\n' \
                        .format(i, action, env.world.current_date)
            else:
                sub_menu_actions += '| {0:2}) {1:25} | {2:25}|\n' \
                    .format(i, action, ' ')
            i += 1
        sub_menu_actions += \
            '+-------------------------------+--------------------------+\n'

        # add main menu tag
        menu = \
            '+----------------------------------------------------------+\n' \
            '|                       Testing menu                       |\n' \
            '|----------------------------------------------------------|\n' \
            '{0}'.format(sub_menu_actions, 'Value:', 'Previous:', '?',
                         'Current')

        render_menu = 'Rendered values:\n'
        render_menu += \
            '+---------------------------+-----------+---+---------+\n'
        render_menu += '| {0:25} | {1:10}| {2} | {3:8}|\n'.format('Value:',
                                                                  'Previous:',
                                                                  '?',
                                                                  'Current')
        render_menu += \
            '+---------------------------+-----------+---+---------+\n'
        for i in range(len(last_render[0])):
            if float(last_render[1][i]) < float(curr_render[1][i]):
                mark = '<'
            elif float(last_render[1][i]) > float(curr_render[1][i]):
                mark = '>'
            else:
                mark = '='

            render_menu += '| {0:25} |{1:10.5f} | {2} | {3:7.5f} |\n'.format(
                last_render[0][i], last_render[1][i], mark, curr_render[1][i])
        render_menu += \
            '+---------------------------+-----------+---+---------+\n'

        menu += render_menu

        # print build menu
        print(menu)

        # update lists for plots
        for i in range(len(curr_render[1])):
            values_for_plt[i].append(curr_render[1][i])

        if file_auto_log:
            log_file.write(render_menu)

        # ---------- build menu end ----------

        try:
            option = input('\nSelect option:\n')

            if int(option) in range(1, len(actions) + 1):
                last_render = curr_render

                if file_auto_log:
                    log_file.write(
                        '\nCurrent step: {0}\n'
                        'Chosen action: {1}\n'.format(
                            step + 1, actions[int(option) - 1]))

                # pass the action with the step
                env.step(actions[int(option) - 1])
                curr_render = env.render()
                step += 1
            elif int(option) == len(actions) + 1:
                file_auto_log = not file_auto_log
                if file_auto_log:
                    log_file.write('\n----- New Test ----\n\n')
                    step = 0
                    env.reset()
                    last_render = curr_render = env.render()
                    for i in values_for_plt:
                        i.clear()
            elif int(option) == len(actions) + 2:
                for i in range(len(curr_render[0])):
                    plt.plot(values_for_plt[i], label=curr_render[0][i])
                plt.legend()
                plt.show()
            elif int(option) == len(actions) + 3:
                time = float(input('Pass time in hour:\n'))
                while (time - env.world.time_step_in_minutes/60) >= 0:

                    last_render = curr_render

                    # pass the action with the step
                    env.step('action_nop')

                    curr_render = env.render()
                    step += 1

                    # update lists for plots
                    for i in range(len(curr_render[1])):
                        values_for_plt[i].append(curr_render[1][i])

                    time -= env.world.time_step_in_minutes/60

            elif int(option) == len(actions) + 4:
                break
            else:
                raise ValueError()
        except ValueError:
            print("Invalid option!")

    # while end, close file and save logs
    log_file.close()


def main():
    """Run the experiment and save the results if needed"""

    # TODO: properly implement me (after environment and agent get done)
    save_experiment = False

    # start manual tests
    run_manual_tests = True
    if run_manual_tests:
        manual_testing()

    env = HouseEnergyEnvironment()
    # env.render() ?
    # params = dict() ?
    agent = Agent(env=env)
    # agent.network = load_model("models/model1.xxx") ?

    rewards = []
    num_episodes = 10000
    for i in range(num_episodes):
        print("episode {} / {}".format(i, num_episodes))
        rewards.append(agent.run())

    plt.plot(rewards)
    plt.show()

    # after learning
    # recover any important info about env, agent etc.
    info = None
    if save_experiment:
        # save to database
        save_to_database(info)


if __name__ == "__main__":
    main()
