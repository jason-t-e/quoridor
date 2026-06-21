from .strategy_guide import STRATEGY_RULES

def compute_agreement_scores(board, action, rules=STRATEGY_RULES):
    return [
        (rule.name, rule.agreement(board, action))
        for rule in rules if rule.condition(board)
    ]

def total_agreement_bonus(scores, weights):
    return sum(weights.get(name, 1.0) * score for name, score in scores)
