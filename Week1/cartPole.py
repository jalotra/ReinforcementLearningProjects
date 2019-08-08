import numpy as np
import gym

def play(env, policy):
    observation = env.reset()

    done = False
    score = 0
    observations = []
    

    def mainLoop(observation, done, score):
        for _ in range(5000):
            observations = list(observation) 

            if done:
                break

            outcome = np.dot(observations, np.dot(1, policy).flatten()) 

            if outcome > 0:
                action = 1
            else:
                action = 0
            
            observation, reward, done, info = env.step(action)
            score += reward

        return score, observations
    
    return mainLoop(observation, done, score)

def Epochs(number_of_epochs, env, policy):
    main_score = []
    for _ in range(number_of_epochs):
        main_score.append(play(env, policy)[0])
    return main_score

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    CurrentPolicy = np.random.randn(1, 4)
    print('Policy Score' , Epochs(100, env, CurrentPolicy))
