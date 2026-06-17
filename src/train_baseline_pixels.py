from configs import baseline_pixels

import arc_agi
import time

def get_env_names():
    arcade = arc_agi.Arcade()
    available_games = arcade.get_environments()

    env_names = []
    for env_id in range(len(available_games)):
        env_name = str(available_games[env_id].game_id)
        env_names.append(env_name)

    return env_names

if __name__ == "__main__":
    num_steps = 10000

    env_names = get_env_names()

    print("available envs ", env_names)
    print("num       envs ", len(env_names))

    config = baseline_pixels.BaselinePixelsConfig(env_names[5])

    # training loop
    fps = 0
    for n in range(num_steps):
        time_start = time.time()
        result_logs = config.agent.learning_step()
        time_stop = time.time()

        fps = 0.9*fps + 0.1*1.0/(time_stop - time_start)

        print(fps)


    '''
    num_envs = 1
    
    # create wrapper
    envs               = agi_libs.ARCEnvsWrapper(num_envs, [5])

    # reset envs
    available_actions = []
    for n in range(len(envs)):

        obs, info = envs.reset(n)

        available_actions.append(info["available_actions"])

    # custom features extractor
    features_extractor = agi_libs.PixelsFeatureExtractor()

    # create model
    model = agi_libs.models.ModelRawPixels(features_extractor.shape, len(available_actions[0]), 16)

    
    agent = PPOAgent(envs, features_extractor, model, hyperparameters)


    # training loop
    for n in range(num_steps):
        result_logs = agent.learing_step()



    cnt = 0
    while True:
        cnt+= 1
        actions = []
        for j in range(num_envs):
            a_main  = numpy.random.randint(0, len(available_actions[j]))
            ax      = numpy.random.randint(0, 15)
            ay      = numpy.random.randint(0, 15)

            #print(available_actions[j], a_main, ax, ay)

            actions.append((a_main, ax, ay))
        
        env_results = envs.step(actions)


        #obs = features_extractor(env_results[0][0][0])
        #print(obs)


        for j in range(num_envs):
            if env_results[j][2]:
                obs, info = envs.reset(j)

                available_actions[j] = info["available_actions"]

        reward = env_results[0][1]
        if reward != 0:
            print("reward = ", reward)

        if cnt%16 == 0:
            envs.render(0)

    '''