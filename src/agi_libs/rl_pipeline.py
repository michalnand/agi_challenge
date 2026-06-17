



def raw_pixels_baseline(config):
    
    envs = ARCEnvsWrapper(num_envs, env_name)
    envs = FeatureExtractor(envs)

    agent = PPOBaseline(envs)


    for n in range(num_training_steps):
        agent.step()