import rl_libs

import numpy


if __name__ == "__main__":

    force_envs = []
    force_envs.append("bp35")
    #force_envs.append("sc25")
    envs = rl_libs.ARCEnvsWrapper(10, force_envs)
  

    # reset envs
    obs_all  = []
    info_all = []
    for n in range(len(envs)):
        obs, info = envs.reset(n)

        obs_all.append(obs)
        info_all.append(info)

    print("reset result")
    for n in range(len(envs)):
        
        print("obs  ", obs_all[n].shape)
        print("info ", info_all[n])
        print("\n\n\n")
    print("\n\n\n")

    for n in range(100):

        print("running step ", n)

        actions = []
        for j in range(len(envs)):
            action_id = numpy.random.randint(0, len(info_all[j]["available_actions"]))
            x = numpy.random.randint(0, 15)
            y = numpy.random.randint(0, 15)

            actions.append((action_id, x, y))

        # step all envs at once
        # returns list of (obs, reward, done, infos)
        results = envs.step(actions)

        '''
        for j in range(len(envs)):
            obs, reward, done, info = results[j]
            print(obs.shape, reward, done, info)

        print("\n\n\n\n\n")
        '''

        for j in range(len(envs)):
            obs, reward, done, info = results[j]
            if done:
                envs.reset(j)
                print("reset ", n, j, results[j][3])

        envs.render(0)


    '''
  
    actions = envs.get_actions(0)


    print(envs.get_actions(0))


    res = envs.step_one(0, actions[0])
    print("\n\n\n")
    print("step")
    print(res)
    '''