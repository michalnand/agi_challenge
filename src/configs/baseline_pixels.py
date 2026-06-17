import agi_libs

class BaselinePixelsConfig:

    def __init__(self, env_name):
        self.num_envs  = 128
        self.num_steps = 1000000

        # create envs
        envs                = agi_libs.ARCEnvsWrapper(self.num_envs, [env_name])

        # create instan features extractor
        features_extractor  = agi_libs.PixelsFeatureExtractor()

        # env reset to obtain action space
        obs, info = envs.reset(0)
        num_actions   = len(info["available_actions"])
        num_actions_x = envs.envs[0].num_actions_x
        

        # init model
        self.model = agi_libs.models.ModelRawPixels(features_extractor.shape, num_actions, num_actions_x)

        hyperparameters = {}
        hyperparameters["learning_rate"] = 0.0001

        # instantiate agent
        self.agent = agi_libs.PPOAgent(envs, features_extractor, self.model, hyperparameters)

        