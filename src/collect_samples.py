import agi_libs

import numpy

import arc_agi





if __name__ == "__main__":

    arcade = arc_agi.Arcade()
    available_games = arcade.get_environments()

    print("total games ", len(available_games))

    num_steps = 4096

    result_path = "data/states_samples/"
    

    for env_id in range(len(available_games)):
        env_name = str(available_games[env_id].game_id)

        print("collecting from ", env_id, env_name)

        env = agi_libs.EnvWrapper(arcade, env_name)

        obs, info = env.reset() 

        obs_result = []

        for n in range(num_steps):
            action_id = numpy.random.randint(0, len(info["available_actions"]))
            x = numpy.random.randint(0, 15)
            y = numpy.random.randint(0, 15)
        
            obs, reward, done, info = env.step((action_id, x, y))

            if done:
                obs, info = env.reset()

            

            obs_result.append(obs[0])

        obs_result = numpy.array(obs_result)
        obs_result = numpy.array(obs_result, dtype=numpy.uint8)
        

        file_name = result_path + env_name + ".npy"
        print("saving ", obs_result.shape, " to ", file_name)
        numpy.save(file_name, obs_result)

