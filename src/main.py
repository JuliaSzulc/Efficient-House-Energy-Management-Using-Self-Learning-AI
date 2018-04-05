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
        last_render TODO: depends on what env.render() returns
        log_file (file) - output for logs

    """

    env = HouseEnergyEnvironment()
    actions = env.get_actions()

    step = 0
    file_auto_log = False
    last_render = env.render()
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
                    '| {0:2}) {1:25} | {2:2}) Exit tests {3:10}|\n' \
                        .format(i, action, j, ' ')
                j += 1
            elif i == 3:
                sub_menu_actions += \
                    '| {0:2}) {1:25} |--------------------------|\n' \
                        .format(i, action)
            elif i == 4:
                sub_menu_actions += \
                    '| {0:2}) {1:25} | Current step: {3:10} |\n' \
                        .format(i, action, ' ', step)
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
            '{0}Last values:\n {1}\nCurrent values:\n {2}'.format(
                sub_menu_actions, last_render,
                env.render())

        # print build menu
        print(menu)

        # ---------- build menu end ----------

        try:
            option = input('\nSelect option:\n')

            if int(option) in range(1, len(actions) + 1):
                last_render = env.render()

                if file_auto_log:
                    log_file.write(
                        'Current step: {0}\n'
                        'Last values:{1}\2'
                        'Current values:{2}\n'
                        'Chosen action:{3}\n'.format(
                            step, last_render, env.render(),
                            actions[int(option) - 1]))

                # pass the action with the step
                env.step(actions[int(option) - 1])
                step += 1
            elif int(option) == len(actions) + 1:
                file_auto_log = not file_auto_log
                if file_auto_log:
                    log_file.write('\n----- New Test ----\n\n')

            elif int(option) == len(actions) + 2:
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
