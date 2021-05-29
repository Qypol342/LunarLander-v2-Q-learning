#importation de librairie publique
import numpy as np
from utils import plotLearning
import gym
# importation de ma libraire agent
from DQN import Agent ,Agent_Replay

if __name__ == '__main__':
	#création de l'environnement
	env = gym.make('LunarLander-v2')
	#nombre d’épisode
	n_games = 200
	#création d’un agent
	agent = Agent(gamma=0.99, epsilon=0.0, alpha=0.0005, input_dims=8,
		n_actions=4, mem_size=10000000, batch_size=64, epsilon_end=0.00,update_target_network =100)
	

	# enlever le # pour charger un model
	agent.load_model()

	#grader un historique des scores et de epsilon pour fair un graph
	scores = []
	eps_history = []
	
	#mettre False pour faire l’entrainement sans visuel
	render = True

	for i in range(n_games):
		# initialization / reinitialisation
		done = False
		score = 0
		observation = env.reset()
		while not done:
			# calculer l’action de l’agent
			action = agent.choose_action(observation)
			# réaliser l’action
			# récupère les observation pour la prochaine étape
			# récupère le score et le statut de la partie
			observation_, reward, done, info = env.step(action)
			score += reward
			# la class agent sauvegarde les variables
			#agent.remember(observation, action, reward, observation_, done)
			observation = observation_
			# met à jour le tableau / dans notre cas le réseau de neurone
			#agent.learn()
			if render:
				still_open = env.render()
				if still_open == False: break

		# enregistre les résultats 
		eps_history.append(agent.epsilon)
		scores.append(score)

		# calcule est affichage des scores
		avg_score = np.mean(scores[max(0,i-100):(i+1)])
		print('episode',i,"score %.2f" % score,
				'avergae score %.2f' % avg_score)

		
		# sauvegarder le modèle tout les 10 épisodes
		if i % 10 ==0 and i > 0:
			agent.save_model(str(score))
	
	# afficher le graphique
	filename = 'lunarlander.png'
	x = [i+1 for i in range(n_games)]
	plotLearning(x,scores, eps_history, filename)


