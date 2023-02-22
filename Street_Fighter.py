import retro 
import numpy as np
import cv2
import neat
import pickle
import time
Game = "StreetFighterIISpecialChampionEdition-Genesis"
env = retro.make(Game)
imgarray = []
def eval_genomes(genomes, config): 
	for genome_id, genome in genomes:
		ob = env.reset()
		ac = env.action_space.sample()
		iny, inx, inc = env.observation_space.shape
		inx = int(inx/8)
		iny = int(iny/8)
		net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

		fitness_current = 0
		health = 176
		healthTracker = 176
		enemy_health = 176
		enemy_healthTracker = 176
		enemy_matches_won = 0
		enemy_matches_wonTracker = 0
		matches_won = 0
		matches_wonTracker = 0
		done = False

		while not done:
			env.render()
			ob = cv2.resize(ob, (inx, iny))
			ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
			ob = np.reshape(ob, (inx, iny))
			imgarray = ob.flatten()
			nnOutput = net.activate(imgarray)
			ob, rew, done, info = env.step(nnOutput)        
			enemy_health = info['enemy_health']
			health = info['health']
			enemy_matches_won = info['enemy_matches_won']
			matches_won = info ['matches_won']			
			if health < healthTracker:	
				fitness_current += (health - healthTracker)
				healthTracker = health
			elif health > healthTracker:
				healthTracker = health
		
			if enemy_health < enemy_healthTracker:
				fitness_current += (enemy_healthTracker - enemy_health)
				enemy_healthTracker = enemy_health
			elif enemy_health > enemy_healthTracker:
				enemy_healthTracker = enemy_health
		
			if matches_won > matches_wonTracker:
				matches_wonTracker = matches_won
				fitness_current += 1000
			elif matches_won < matches_wonTracker:
				matches_wonTracker = matches_won
				fitness_current += 2000
				health = 176
				enemy_health = 176

			if enemy_matches_won > enemy_matches_wonTracker:
				enemy_matches_wonTracker = enemy_matches_won
				fitness_current += -1000
				healthTracker = 176
				enemy_healthTracker = 176
			elif enemy_matches_won < enemy_matches_wonTracker:
				enemy_matches_wonTracker = enemy_matches_won
	
		if done == True:
			print(genome_id, fitness_current)
			env.reset()
				
			genome.fitness = fitness_current
			
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, 
neat.DefaultStagnation, 'config-feedforward-SF2')
p = neat.Population(config)
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-0')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))
winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
	pickle.dump(winner, output, 1)
