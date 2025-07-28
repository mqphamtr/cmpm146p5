import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# The level as a grid of tiles


class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        
        # simple coefficients - focus on core quality
        coefficients = dict(
            meaningfulJumpVariance=0.3,
            negativeSpace=0.6,
            pathPercentage=0.3,
            emptyPercentage=0.5,
            linearity=-0.3,
            solvability=2
        )
        
        base_fitness = sum(map(lambda m: coefficients[m] * measurements[m], coefficients))
        
        # just a few simple adjustments
        level = self.to_level()
        adjustments = 0
        
        # penalty -> floating enemies (simple check)
        floating_enemies = 0
        for y in range(len(level) - 1):
            for x in range(1, len(level[0]) - 1):
                if level[y][x] == 'E':
                    if level[y + 1][x] not in ['X', '?', 'B']:
                        floating_enemies += 1
        
        adjustments -= floating_enemies * 0.1
        
        # penalty -> broken pipes (simple check)
        broken_pipes = 0
        for y in range(len(level)):
            for x in range(1, len(level[0]) - 1):
                if level[y][x] == 'T':  # pipe top
                    if y + 1 < len(level) and level[y + 1][x] != '|':
                        broken_pipes += 1
        
        adjustments -= broken_pipes * 0.15
        
        # bonus -> having some coins and enemies
        coin_count = sum(row.count('o') for row in level)
        enemy_count = sum(row.count('E') for row in level)
        
        if coin_count >= 3:
            adjustments += 0.2
        if 2 <= enemy_count <= 8:
            adjustments += 0.2
        
        self._fitness = base_fitness + adjustments
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # mutate a genome into a new genome - this is where we make changes to evolve levels
    def mutate(self, genome):
        mutated_genome = copy.deepcopy(genome)
        
        # reduced these rates so we don't destroy structures faster than we build them
        tile_flip_rate = 0.02      # 2% chance per tile - was destroying platforms before
        structure_add_rate = 0.08   # 8% chance to add new structures - increased to build more  
        structure_remove_rate = 0.02 # 2% chance to clean up floating blocks - reduced to preserve
        enemy_adjust_rate = 0.04    # 4% chance to fix enemy placement
        
        # type 1 -> individual tile mutations (but less destructive now)
        for y in range(1, height - 2):  # skip top and bottom rows
            for x in range(1, width - 1):  # skip first and last columns
                if random.random() < tile_flip_rate:
                    current_tile = mutated_genome[y][x]
                    
                    # if empty space, maybe add something but rarely question marks
                    if current_tile == "-": 
                        new_tile = random.choices(["-", "o", "X", "?", "B"], 
                                                weights=[0.85, 0.1, 0.03, 0.01, 0.01])[0]  # way less ? blocks
                        mutated_genome[y][x] = new_tile
                    
                    # if solid block, less likely to remove it (preserve platforms)
                    elif current_tile in ["X", "?", "B"]:
                        new_tile = random.choices(["X", "?", "B", "-"], 
                                                weights=[0.6, 0.15, 0.15, 0.1])[0]  # only 10% chance to remove
                        mutated_genome[y][x] = new_tile
                    
                    # coins can be removed sometimes
                    elif current_tile == "o":
                        if random.random() < 0.3:
                            mutated_genome[y][x] = "-"
        
        # type 2 -> add new structures (biased toward platforms)
        if random.random() < structure_add_rate:
            # heavily favor platforms since we need more of them
            structure_type = random.choices(
                ["platform", "pipe", "coin_line", "block_cluster"],
                weights=[0.5, 0.2, 0.1, 0.2]  # 50% chance of platform
            )[0]
            
            if structure_type == "platform":
                plat_x = random.randint(30, width - 50)  # more centered placement
                plat_y = random.randint(8, 13)           # reasonable height
                plat_width = random.randint(3, 8)        # bigger platforms
                plat_material = random.choices(["X", "?", "B"], 
                                            weights=[0.8, 0.1, 0.1])[0]  # mostly solid blocks
                
                # less strict about where we can place - allow some overlap
                occupied_count = 0
                for check_x in range(plat_x, min(plat_x + plat_width, width - 1)):
                    if mutated_genome[plat_y][check_x] != "-":
                        occupied_count += 1
                
                # place platform if less than half the area is already occupied
                if occupied_count < plat_width // 2:
                    for i in range(plat_width):
                        if plat_x + i < width - 1:
                            mutated_genome[plat_y][plat_x + i] = plat_material
            
            # todo: add other structure types here when needed
        
        # type 3 -> remove floating structures (but be gentle about it)
        if random.random() < structure_remove_rate:
            # only remove blocks that are completely isolated
            for y in range(height - 3, 0, -1):  # work top to bottom, skip ground area
                for x in range(1, width - 1):
                    if mutated_genome[y][x] in ["X", "?", "B"]:
                        # check 3x3 area for support instead of just directly below
                        has_support = False
                        for check_x in range(max(0, x-1), min(width, x+2)):
                            if y + 1 < height and mutated_genome[y + 1][check_x] in ["X", "?", "B", "|"]:
                                has_support = True
                                break
                        
                        # only remove if really floating and random chance
                        if not has_support and random.random() < 0.3:
                            mutated_genome[y][x] = "-"
        
        # type 4 -> fix enemy placement (move floating enemies to solid ground)
        if random.random() < enemy_adjust_rate:
            for y in range(height - 1):
                for x in range(1, width - 1):
                    if mutated_genome[y][x] == "E":
                        # check if enemy is floating
                        if y + 1 >= height or mutated_genome[y + 1][x] not in ["X", "?", "B"]:
                            mutated_genome[y][x] = "-"  # remove floating enemy
                            
                            # try to place it on solid ground nearby
                            for try_x in range(max(1, x - 10), min(width - 1, x + 10)):
                                ground_y = height - 2  # one tile above ground
                                if (mutated_genome[ground_y][try_x] == "-" and 
                                    mutated_genome[ground_y + 1][try_x] == "X"):
                                    mutated_genome[ground_y][try_x] = "E"
                                    break  # found a spot, stop looking
        
        return mutated_genome

    # Create zero or more children from self and other
    def generate_children(self, other):
        new_genome = copy.deepcopy(self.genome)
        
        # column-wise crossover: pick a crossover point
        crossover_point = random.randint(10, width - 10)
        
        # take everything from crossover_point onward from other parent
        for y in range(height):
            for x in range(crossover_point, width - 1):  # skip last column (flag)
                new_genome[y][x] = other.genome[y][x]
        
        new_genome = self.mutate(new_genome)
        return (Individual_Grid(new_genome),)
    

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    
    @classmethod
    def random_individual(cls):
        # start with completely empty level
        g = [["-" for col in range(width)] for row in range(height)]
        
        # create mostly solid ground with some holes for variety
        for x in range(width):
            if random.random() < 0.9:  # 90% chance of solid ground, 10% holes
                g[15][x] = "X"
        
        # create platform clusters instead of scattered random platforms
        num_clusters = random.randint(2, 4)  # 2-4 groups of platforms
        
        for cluster in range(num_clusters):
            # pick center point for this cluster
            cluster_center = random.randint(40, width - 60)  # avoid edges
            platforms_in_cluster = random.randint(2, 5)      # 2-5 platforms per cluster
            
            for _ in range(platforms_in_cluster):
                # place platforms within 30 tiles of cluster center
                platform_x = cluster_center + random.randint(-30, 30)
                platform_x = max(10, min(platform_x, width - 20))  # keep in bounds
                
                platform_y = random.randint(8, 13)    # reasonable jumping height
                platform_width = random.randint(3, 8)  # decent size platforms
                platform_type = random.choices(["X", "?", "B"], weights=[0.8, 0.1, 0.1])[0]  # mostly solid
                
                # actually place the platform tiles
                for x in range(platform_x, min(platform_x + platform_width, width - 1)):
                    g[platform_y][x] = platform_type
        
        # add some pipes for vertical variety
        num_pipes = random.randint(1, 3)
        
        for _ in range(num_pipes):
            pipe_x = random.randint(20, width - 40)  # give space around pipes
            pipe_height = random.randint(2, 6)       # reasonable pipe heights
            
            # make sure pipe has solid base
            g[15][pipe_x] = "X"
            g[15][pipe_x + 1] = "X"
            
            # build the pipe body from bottom up
            for y in range(15 - pipe_height, 15):
                g[y][pipe_x] = "|"
            
            # add pipe top
            g[15 - pipe_height - 1][pipe_x] = "T"
        
        # add coins but only where mario can actually reach them
        for _ in range(random.randint(1, 6)):  # reduced from 3-12 to avoid coin spam
            coin_x = random.randint(1, width - 2)
            coin_y = random.randint(5, 14)
            
            # check if coin is reachable (platform within jumping distance below)
            can_reach = False
            for check_y in range(coin_y + 1, min(coin_y + 5, height)):
                if g[check_y][coin_x] in ["X", "?", "B", "T"]:
                    can_reach = True
                    break
            
            # only place coin if mario can get it
            if can_reach and g[coin_y][coin_x] == "-":
                g[coin_y][coin_x] = "o"
        
        # place enemies on solid ground only
        for _ in range(random.randint(2, 6)):
            enemy_x = random.randint(1, width - 2)
            
            # find solid surface from bottom up
            for y in range(14, -1, -1):  # work from bottom up
                if g[y + 1][enemy_x] in ["X", "?", "B"] and g[y][enemy_x] == "-":
                    g[y][enemy_x] = "E"
                    break  # only place one enemy per x position
        
        # add all the required mario level elements
        g[15][:] = ["X"] * width        # ensure solid bottom row
        g[14][0] = "m"                  # mario start position
        g[7][-1] = "v"                  # goal flagpole
        # goal flag
        for col in range(8, 14):
            g[col][-1] = "f"
        # solid ground at end
        for col in range(14, 16):                   
            g[col][-1] = "X"
        
        return cls(g)

    @classmethod
    def empty_individual(cls):
        #here we create the minimum playable level, so we know that it actually is runnable
        g = [["-" for col in range(width)] for row in range(height)]
        
        #flat level with solid ground
        g[15][:] = ["X"] * width
        
        #basic platforms for variety
        for x in range(50, 58):
            g[13][x] = "X"
        for x in range(100, 106):
            g[11][x] = "X"
        for x in range(150, 155):
            g[9][x] = "X"
        
        # coins and enemies
        g[12][52] = "o" 
        g[10][102] = "o" 
        g[14][30] = "E"  
        g[12][103] = "E"  
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 2
        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            print(metrics.metrics(self.to_level()))
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things
        if random.random() < 0.1 and len(new_genome) > 0:
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            new_de = de
            x = de[0]
            de_type = de[1]
            choice = random.random()
            if de_type == "4_block":
                y = de[2]
                breakable = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    breakable = not de[3]
                new_de = (x, de_type, y, breakable)
            elif de_type == "5_qblock":
                y = de[2]
                has_powerup = de[3]  # boolean
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    has_powerup = not de[3]
                new_de = (x, de_type, y, has_powerup)
            elif de_type == "3_coin":
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    h = offset_by_upto(h, 2, min=2, max=height - 4)
                new_de = (x, de_type, h)
            elif de_type == "0_hole":
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                h = de[2]
                dx = de[3]  # -1 or 1
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    h = offset_by_upto(h, 8, min=1, max=height - 4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, dx)
            elif de_type == "1_platform":
                w = de[2]
                y = de[3]
                madeof = de[4]  # from "?", "X", "B"
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=width - 2)
                elif choice < 0.75:
                    y = offset_by_upto(y, height, min=0, max=height - 1)
                else:
                    madeof = random.choice(["?", "X", "B"])
                new_de = (x, de_type, w, y, madeof)
            elif de_type == "2_enemy":
                pass
            new_genome.pop(to_change)
            heapq.heappush(new_genome, new_de)
        return new_genome

    def generate_children(self, other):
        # STUDENT How does this work?  Explain it in your writeup.
        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)


Individual = Individual_Grid


def generate_successors(population):
    results = []
    # STUDENT Design and implement this
    # Hint: Call generate_children() on some individuals and fill up results.

    # Elitism: carry over top N individuals
    elite_count = int(len(population) * 0.2)
    sorted_pop = sorted(population, key=lambda ind: ind.fitness(), reverse=True)
    results.extend(sorted_pop[:elite_count])

    # Tournament selection for the rest
    def tournament(k=5):
        competitors = random.sample(population, k)
        return max(competitors, key=lambda ind: ind.fitness())

    while len(results) < len(population):
        parent1 = tournament()
        parent2 = tournament()
        children = parent1.generate_children(parent2)
        results.extend(children)

    results = results[:len(population)]

    return results

def add_padding_columns(level, num_cols=3):
    height = len(level)
    for row in range(height):
        if row == height - 1:
            level[row].extend(["X"] * num_cols)  # solid ground
        else:
            level[row].extend(["-"] * num_cols)  # empty air


def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        level = best.to_level()
                        add_padding_columns(level, 3)
                        for row in level:
                            f.write("".join(row) + "\n")

                generation += 1
                # STUDENT Determine stopping condition
                stop_condition = False
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            level = final_gen[k].to_level()
            add_padding_columns(level, 3)
            for row in level:
                f.write("".join(row) + "\n")