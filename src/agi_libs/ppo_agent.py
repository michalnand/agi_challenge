import numpy

class PPOAgent:
    def __init__(self, envs, features_extractor, model, hyperparameters):
        print("initializing PPO agent")

        self.envs = envs
        self.features_extractor = features_extractor

        self.num_envs = len(envs)
        self.num_actions = len(envs.envs[0].action_space)
        self.num_actions_x = envs.envs[0].num_actions_x
        self.num_actions_y = envs.envs[0].num_actions_y

        print("self.num_actions ", self.num_actions)

        # reset env
        env_results = []
        for n in range(self.num_envs):
            obs, info = self.envs.reset(n)
            env_results.append([obs])

        self.state = self._get_observation(env_results)

        print("env reset done, initial state shape ", self.state.shape)


    def learning_step(self):
        
        # TODO
        # obtain action from agent
        
        # random actions
        actions_main = numpy.random.randint(0, self.num_actions, (self.num_envs))
        actions_x = numpy.random.randint(0, self.num_actions_x, (self.num_envs))
        actions_y = numpy.random.randint(0, self.num_actions_y, (self.num_envs))

        actions = numpy.stack([actions_main, actions_x, actions_y])
        actions = actions.T

        #print("actions = ", actions.shape)

        # all envs parallel step
        env_results = self.envs.step(actions)

        # episode done, reset
        for n in range(self.num_envs):
            if env_results[n][2]:
                obs, info = self.envs.reset(n)
                env_results[n][0] = obs


        # feature engineering for observaiton
        self.state = self._get_observation(env_results)





    def _get_observation(self, raw_obs):

        result = []
        for n in range(len(raw_obs)):
            result.append(self.features_extractor(raw_obs[n][0][0]))

        result = numpy.stack(result)

        return result
     