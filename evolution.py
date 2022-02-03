import copy

from player import Player

class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # # Top-k algorithm
        # players.sort(key=lambda p: p.fitness, reverse=True) 
        # survivors = players[:num_players]  
        # survivors.sort(key=lambda p: p.fitness, reverse=True)    
        # return players[: num_players]
        
        # # Roulette wheel
        # players.sort(key=lambda p: p.fitness, reverse=True)
        # survivors = self.RW(players,num_players)
        # survivors.sort(key=lambda p: p.fitness, reverse=True)

        # SUS
        players.sort(key=lambda p: p.fitness, reverse=True)
        survivors = self.SUS(players,num_players)
        survivors.sort(key=lambda p: p.fitness, reverse=True)

        # TODO (Additional: Learning curve)
        f = open('result.txt', 'a')
        f.write("{} {} {}\n".format(str(survivors[0].fitness),str(survivors[-1].fitness),statistics.mean([p.fitness for p in players])))
        f.close()

        return survivors

    def RW(self, players, num_players):
        sum_of_fitnesses = sum([player.fitness for player in players])
        p_list = [(player,player.fitness/sum_of_fitnesses) for player in players]
        
        survivors = list()
        while(len(survivors)<num_players):
            ur = random.uniform(0,1)
            start, end = 0, 0
            for p in p_list:
                end = start + p[1]
                if (start < ur) and (ur < end):
                    survivors.append(p[0])
                    break
                else:
                    start = end
        survivors.sort(key=lambda s: s.fitness, reverse=True)

        return survivors


    def SUS(self, players, num_players):

        sum_of_fitnesses = sum([player.fitness for player in players])
        p_list = [(player,player.fitness/sum_of_fitnesses) for player in players]
        ruler_size = 1 - 1/num_players
        ruler = np.linspace(0.0, ruler_size, num_players)
        ur = random.uniform(0,1/num_players)       
        new_ruler = [i+ur for i in ruler]

        survivors = list()

        start, end = 0, 0
        for p in p_list:
            end = start + p[1]            
            for r in new_ruler:
                if (start < r) and (r < end):
                    survivors.append(p[0])
            start = end

        survivors.sort(key=lambda s: s.fitness, reverse=True)
        return survivors
        

    def Q_toournament(self):
        pass

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """

        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            prev_players.sort(key=lambda p: p.fitness, reverse=True)

            pairs = [tuple(self.SUS(prev_players[:round(len(prev_players)/2)],2)) for i in range(0,num_players,2)]

            new_players = list()
            for pair in pairs:
                child_1, child_2 = self.crossover(self.clone_player(pair[0]),self.clone_player(pair[1]))
                child_1, child_2 = self.mutation(child_1,child_2)

                new_players = new_players + [child_1,child_2]
            # new_players = prev_players
            return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def crossover(self, parent_1, parent_2):

        layer_sizes = parent_1.nn.layer_sizes = parent_2.nn.layer_sizes
        hl_p1 = parent_1.nn.layers_parameters
        hl_p2 = parent_1.nn.layers_parameters

        
        for layer in range(len(layer_sizes)-1):

            current_layer_size = layer_sizes[layer]
            next_layer_size = layer_sizes[layer+1]

            w_p1 = hl_p1[layer][0]
            w_p2 = hl_p2[layer][0]

            num_of_candidates = random.choice(range(0, next_layer_size))
            candidates_indexes = random.choices(range(0,next_layer_size), k=num_of_candidates)

            for cc in range(0,num_of_candidates):

                chromosome_p1 = w_p1[candidates_indexes[cc]]
                chromosome_p2 = w_p2[candidates_indexes[cc]]

                divider_index = random.choice(range(0, current_layer_size))
                # print(chromosome_p1,chromosome_p2,divider_index)

                chromosome_c1 = np.concatenate((chromosome_p1[:divider_index], chromosome_p2[divider_index:]))
                chromosome_c2 = np.concatenate((chromosome_p2[:divider_index], chromosome_p1[divider_index:]))
                # print(chromosome_c1,chromosome_c2,divider_index)

                w_p1[candidates_indexes[cc]] = chromosome_c1
                w_p2[candidates_indexes[cc]] = chromosome_c2

        return parent_1,parent_2

        

    def mutation(self,parent_1,parent_2):

        layer_sizes = parent_1.nn.layer_sizes = parent_2.nn.layer_sizes
        hl_p1 = parent_1.nn.layers_parameters
        hl_p2 = parent_1.nn.layers_parameters

        for layer in range(len(layer_sizes)-1):

            current_layer_size = layer_sizes[layer]
            next_layer_size = layer_sizes[layer+1]

            w_p1 = hl_p1[layer][0]
            w_p2 = hl_p2[layer][0]

            for row in range(0,next_layer_size):
                for col in range(0,current_layer_size):
                    ur = random.uniform(0, 1)
                    if ur < 0.4:
                        w_p1[row][col] = random.uniform(0, 1)

        return parent_1,parent_2
