# elo.py
from math import pow

class Elo:
    def __init__(self, base=1500, k=20):
        self.ratings = {}
        self.base = base
        self.k = k

    def get(self, team):
        return self.ratings.get(team, self.base)

    def expected(self, ra, rb):
        return 1.0 / (1 + pow(10, (rb - ra) / 400.0))

    def update(self, team_a, team_b, score_a):
        # score_a is 1.0 if A wins, 0.0 if loses (no draws in T20)
        ra = self.get(team_a)
        rb = self.get(team_b)
        ea = self.expected(ra, rb)
        eb = 1 - ea
        self.ratings[team_a] = ra + self.k * (score_a - ea)
        self.ratings[team_b] = rb + self.k * ((1 - score_a) - eb)
