import pygame
import os
import random
import neat
pygame.font.init()
pygame.display.set_caption("Infinite Running")

WIDTH, HEIGHT = 750, 500
FPS = 30
GEN = -1
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

main_font = pygame.font.SysFont("arial", 35)
clock = pygame.time.Clock()

class Base:
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def draw(self):
        window = WIN
        pygame.draw.rect(window, self.color, (self.x , self.y, self.width, self.height))

class Player(Base):
    def __init__(self, x, y, width, height, color):
        super().__init__(x, y, width, height, color)
        self.jumping = False
        self.y_gravity = 1
        self.jump_height = 15
        self.y_velocity = self.jump_height
        
    def move(self, player_vel):
        self.x += player_vel

class Floor(Base):
    def __init__(self, x, y, width, height, color):
        super().__init__(x, y, width, height, color)

    def draw(self):
        window = WIN
        pygame.draw.rect(window, self.color, (self.x , self.y, self.width, self.height), 5)

class DeathSpace(Base):
    def __init__(self, x, y, width, height, color):
        super().__init__(x, y, width, height, color)

def main(genomes, config):
    global GEN
    GEN += 1

    nets = []
    ge = []
    players = []

    player_vel = 5
    player_x = 0
    player_y = HEIGHT * 0.85
    player_width = 40
    player_height = 40
    player_color = (255, 255, 255)

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        players.append(Player(player_x, player_y, player_width, player_height, player_color))
        g.fitness = 0
        ge.append(g)
        
    run = True
    time_survived = 0
    floor = Floor(0, player_y + player_height, WIDTH, HEIGHT, (255, 255, 255))
    death_space = DeathSpace(random.randint(100, 500), 0, random.randint(60, 120), HEIGHT, (0, 0, 0))

    def redraw_window():
        WIN.fill((0, 0, 0))
        floor.draw()
        death_space.draw()

        for x, player in enumerate(players):
            player.draw()
            player.move(player_vel)
            ge[x].fitness += 0.1

            if player.x >= WIDTH - 25 + player.width:
                player.x = player_x
                death_space.x = random.randint(100, 500)
                death_space.width = random.randint(60, 120)

            if player.x + player_width - 5 in range(death_space.x, death_space.x + death_space.width + 1) and player.y == player_y:
                ge[x].fitness -= 5
                players.pop(x)
                nets.pop(x)
                ge.pop(x)
            
            if player.jumping:
                player.y -= player.y_velocity
                player.y_velocity -= player.y_gravity

                if player.y_velocity < -(player.jump_height):
                    player.jumping = False
                    player.y_velocity = player.jump_height

        time_label = main_font.render(f"Seconds Survived: {time_survived/FPS:.2f}", 16, (255, 255, 255))
        gen_label = main_font.render(f"GEN: {GEN}", 16, (255, 255, 255))
        alive_label = main_font.render(f"Alive: {len(players)}", 16, (255, 255, 255))
        WIN.blit(time_label, (WIDTH - time_label.get_width() - 10, 0))
        WIN.blit(gen_label, (0, 0))
        WIN.blit(alive_label, (0, gen_label.get_height() + 10))
        pygame.display.update()

    while run:
        clock.tick(FPS)

        redraw_window()

        if len(players) < 1:
            run = False
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        for x, player in enumerate(players):
            output = nets[x].activate((player.x, abs(player.x - death_space.x), player.y, abs(player.y - death_space.y)))

            if output[0] > 0.5:
                player.jumping = True

        time_survived += 1

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)
