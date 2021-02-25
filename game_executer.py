import gym


def open_game(game_to_play):
	
	# Create environment
	env = gym.make(game_to_play)

	#Execute the environment for a quantity of predefined episodes
	#Each episode is a game itself
	for i_episode in range(20):
		
		#Reset environment to get initial observation
		observation = env.reset()

		#Execute each frame of the game
		for i_frame in range(500):

			#Open window for visualize game
			env.render()

			#Agent takes an action to interact with environment

			#action = env.action_space.sample()

			# Testing implementation of non-random actions:
			action = int(input('Input manual action')) # Push cart to the left or do nothin on rocket


			#Core information for Reinforcement Learning:

			observation, reward, done, info = env.step(action)

			print(f"Observations after {i_frame+1} timesteps:\n")
			print(observation, reward, done, info)
			print(env.action_space)
			print(env.observation_space)

			if done:

				print(f"Episode finished after {i_frame+1} timesteps")
				break

#Game selector:				

print("1: LunarLander-v2, \n2:CartPole-v0")
user_selection = input('Select a game from the list\n')

x = ''
if user_selection == '1':
	x = 'LunarLander-v2'
elif user_selection == '2':
	x='CartPole-v0'

open_game(x)