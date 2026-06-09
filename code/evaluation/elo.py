class EloSystem:
    def __init__(self, k_factor=32):
        self.k_factor = k_factor
        self.ratings = {}

    def get_rating(self, model_id):
        return self.ratings.get(model_id, 1200)

    def set_rating(self, model_id, rating):
        self.ratings[model_id] = rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, model_a, model_b, score_a):
        # score_a: 1 for A win, 0.5 for draw, 0 for A loss
        rating_a = self.get_rating(model_a)
        rating_b = self.get_rating(model_b)

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)

        score_b = 1 - score_a

        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)

        self.set_rating(model_a, new_rating_a)
        self.set_rating(model_b, new_rating_b)

        return new_rating_a, new_rating_b
