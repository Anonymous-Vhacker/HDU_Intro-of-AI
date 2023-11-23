import numpy as np
import random
import copy
import matplotlib.pyplot as plt
 
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
 
def distance(ca, cb):
    dx = abs(ca.x - cb.x)
    dy = abs(ca.y - cb.y)
    distance = np.sqrt((dx ** 2) + (dy ** 2))
    return distance
 
def init_pop(city_list, popSize):
    pop = []
    for i in range(popSize):
        new_city_list = random.sample(city_list, len(city_list))
        pop.append(new_city_list)
 
    return pop
 
def fitness(pop):
    dis_citys = distance_citys(pop)
    return 1.0/dis_citys
 
def distance_citys(pop):
    temp_dis = 0
    for i in range(len(pop)-1):
        temp_dis += distance(pop[i], pop[i+1])
    temp_dis += distance(pop[len(pop)-1], pop[0])
 
    return temp_dis
 
def rank(poplulation):
    rankPop_dic = {}
    for i in range(len(poplulation)):
        fit = fitness(poplulation[i])
        rankPop_dic[i] = fit
 
    return sorted(rankPop_dic.items(), key=lambda x:x[1], reverse=True)
 
 
def select(pop, pop_rank, eliteSize):
    select_pop = []
    for i in range(eliteSize):
        select_pop.append(pop[pop_rank[i][0]])
 
    cumsum = 0
    cumsum_list = []
    temp_pop = copy.deepcopy(pop_rank)
    for i in range(len(temp_pop)):
        cumsum += temp_pop[i][1]
        cumsum_list.append(cumsum)
    for i in range(len(temp_pop)):
        cumsum_list[i] /= cumsum
 
    for i in range(len(temp_pop)-eliteSize):
        rate = random.random()
        for j in range(len(temp_pop)):
            if cumsum_list[j] > rate:
                select_pop.append(pop[pop_rank[i][0]])
                break
 
    return select_pop
 
def breed(pop, eliteSize):
    breed_pop = []
    for i in range(eliteSize):
        breed_pop.append(pop[i])
 
    i = 0
    while i < (len(pop)-eliteSize):
        a = random.randint(0, len(pop)-1)
        b = random.randint(0, len(pop)-1)
        if a != b:
            fa, fb = pop[a], pop[b]
            genea, geneb = random.randint(0, len(pop[a])-1), random.randint(0, len(pop[b])-1)
            startgene = min(genea, geneb)
            endgene = max(genea, geneb)
            child1 = []
            for j in range(startgene, endgene):
                child1.append(fa[j])
            # child1 = copy.deepcopy(fa[:-1])
            child2 = []
            for j in fb:
                if j not in child1:
                    child2.append(j)
            # child2 = [j for j in fb if j not in child1]
            breed_pop.append(child1+child2)
            i = i+1
 
    return breed_pop
 
def mutate(pop, mutationRate):
    mutation_pop = []
    for i in range(len(pop)):
        for j in range(len(pop[i])):
            rate = random.random()
            if rate < mutationRate:
                a = random.randint(0, len(pop[i])-1)
                pop[i][a], pop[i][j] = pop[i][j], pop[i][a]
        mutation_pop.append(pop[i])
 
    return mutation_pop
 
 
def next_pop(population, eliteSize, mutationRate):
    pop_rank = rank(population) #按照适应度排序
    select_pop = select(population, pop_rank, eliteSize) #精英选择策略，加上轮盘赌选择
    breed_pop = breed(select_pop, eliteSize) #繁殖
    next_generation = mutate(breed_pop, mutationRate) #变异
 
    return next_generation
 
#画出路线图的动态变化
def GA_plot_dynamic(city_list, popSize, eliteSize, mutationRate, generations):
    plt.figure('Map')
    plt.ion()
    population = init_pop(city_list, popSize)
 
    print("initial distance:{}".format(1.0/(rank(population)[0][1])))
    for i in range(generations):
        plt.cla()
        population = next_pop(population, eliteSize, mutationRate)
        idx_rank_pop = rank(population)[0][0]
        best_route = population[idx_rank_pop]
        city_x = []
        city_y = []
        for j in range(len(best_route)):
            city = best_route[j]
            city_x.append(city.x)
            city_y.append(city.y)
        city_x.append(best_route[0].x)
        city_y.append(best_route[0].y)
        plt.scatter(city_x, city_y, c='r', marker='*', s=200, alpha=0.5)
        plt.plot(city_x, city_y, "b", ms=20)
        plt.pause(0.1)
 
    plt.ioff()
    plt.show()
 
    print("final distance:{}".format(1.0 / (rank(population)[0][1])))
    bestRouteIndex = rank(population)[0][0]
    bestRoute = population[bestRouteIndex]
    return bestRoute
 
def GA(city_list, popSize, eliteSize, mutationRate, generations):
    population = init_pop(city_list, popSize) #初始化种群
    process = []
 
    print("initial distance:{}".format(1.0/(rank(population)[0][1])))
    for i in range(generations):
        population = next_pop(population, eliteSize, mutationRate) #产生下一代种群
        process.append(1.0 / (rank(population)[0][1]))
 
    plt.figure(1)
    print("final distance:{}".format(1.0 / (rank(population)[0][1])))
    plt.plot(process)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.savefig(str(generations)+ '_' + str(1.0 / (rank(population)[0][1])) + '_' + str(mutationRate) +'_process.png')
 
    plt.figure(2)
    idx_rank_pop = rank(population)[0][0]
    best_route = population[idx_rank_pop]
    city_x = []
    city_y = []
    for j in range(len(best_route)):
        city = best_route[j]
        city_x.append(city.x)
        city_y.append(city.y)
    city_x.append(best_route[0].x)
    city_y.append(best_route[0].y)
    plt.scatter(city_x, city_y, c='r', marker='*', s=200, alpha=0.5)
    plt.plot(city_x, city_y, "b", ms=20)
 
    plt.savefig(str(generations)+'_' + str(mutationRate) + '_route.png')
    plt.show()
 
num_city = 25
city_list = []
 
# for i in range(num_city):
#     x = random.randint(1, 200)
#     y = random.randint(1, 200)
#     city_list.append(City(x, y))

with open('city.txt', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.replace('\n', '')
    # line = line.replace('\t', '')
    city = line.split('\t')
    city_list.append( City( float(city[1]), float(city[2]) ) )
 
# mutationRates = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02]
# for mut in mutationRates:
GA(city_list, 100, 20, 0.01, 1000)