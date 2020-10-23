import numpy as np
import pandas as pd
from typing import Type, Callable
import abc


class Coin():
    def __init__(self, p: float = 0.5):
        self._p = p

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, new_p: float):
        self._p = new_p

    @property
    def entropy(self) -> float:
        return -(
            self.p * np.log(self.p) + (1 - self.p) * np.log(1 - self.p)
        )

    def simulate(self, N: int) -> pd.Series:
        return pd.Series(data=np.random.rand(N) < self.p)


class Bet(abc.ABC):
    """Generic class for bet on the results of generating
    random outcomes of coin.
    Seed parameter is used to consistently recreate the same
    sequence of coin tosses when different Bet subclasses
    use the same seed.
    """
    def __init__(
        self,
        coin: Type[Coin],
        init_fortune: float = 100.0,
        seed: float = 0,
    ):
        self._coin = coin
        self._init_fortune = init_fortune
        self._seed = seed

    @property
    def coin(self) -> Type[Coin]:
        return self._coin

    @coin.setter
    def coin(self, new_coin: Type[Coin]):
        self._coin = new_coin

    @property
    def init_fortune(self) -> float:
        return self._init_fortune

    @init_fortune.setter
    def init_fortune(self, new_init_fortune: float):
        self._init_fortune = new_init_fortune

    @property
    def seed(self) -> float:
        return self._seed

    @seed.setter
    def seed(self, new_seed: float):
        self._seed = new_seed

    @abc.abstractmethod
    def strat(self, results: pd.Series) -> pd.DataFrame:
        """To be instanciated in the children classes.
        Requirements: needs to have three columns, outcomes, bet and size.
        """
        pass

    def simulate(self, N: int) -> pd.DataFrame:
        np.random.seed(self.seed)
        outcomes = self.coin.simulate(N)
        df = self.strat(outcomes)
        df['gains'] = (df['outcomes'] == df['bet']) * df['size'] \
            - (df['outcomes'] != df['bet']) * df['size']
        df['fortune'] = df['gains'].shift(1).cumsum().fillna(0) \
            + self.init_fortune

        return df


class RandomBet(Bet):
    """Random bet head or tails, with a given probability.
    """
    def __init__(
        self,
        coin: Type[Coin],
        p: float = 0.5,
        init_fortune: float = 100.0,
        unit_bet: float = 1.0,
        bet_style: str = 'fractional',
        seed: float = 0,
    ):
        self._p = p
        self._unit_bet = unit_bet
        self._bet_style = bet_style
        super().__init__(
            coin=coin,
            init_fortune=init_fortune,
            seed=seed,
        )

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, new_p: float):
        self._p = new_p

    @property
    def unit_bet(self) -> float:
        return self._unit_bet

    @unit_bet.setter
    def unit_bet(self, new_unit_bet: float):
        self._unit_bet = new_unit_bet

    @property
    def bet_style(self) -> str:
        return self._bet_style

    @bet_style.setter
    def bet_style(self, new_bet_style: str):
        self._bet_style = new_bet_style

    def strat(self, outcomes: pd.Series) -> pd.DataFrame:
        if self.bet_style == 'fractional':
            # self.unit_bet is the fraction to allocate at each round
            df = pd.DataFrame(data=outcomes, columns=['outcomes'])
            bets = np.random.rand(len(outcomes)) < self.p

            sizes = np.empty(len(outcomes))

            fortune = self.init_fortune

            for t, outcome in enumerate(outcomes):
                sizes[t] = self.unit_bet * fortune
                fortune *= 1 + self.unit_bet if bets[t] == outcome\
                    else 1 - self.unit_bet

            df['bet'] = bets
            df['size'] = sizes
        elif self.bet_style == 'incremental':
            # self.unit_bet is the size of the bet
            bets = np.random.rand(len(outcomes)) < self.p
            df = pd.DataFrame(
                data=np.stack(
                    [
                        outcomes,
                        bets,
                        np.ones(len(outcomes)) * self.unit_bet,
                    ],
                    axis=1,
                ),
                columns=['outcomes', 'bet', 'size'],
            )
        else:
            raise ValueError('Unknown bet style {:s}'.format(self.bet_style))
        return df


class ExactArbitraryBet(Bet):
    """User-defined fractional rule.
    """
    def __init__(
        self,
        coin: Type[Coin],
        p: float = 0.5,
        init_fortune: float = 100.0,
        rule: Callable = None,
        seed: float = 0,
    ):
        self._p = p
        self._rule = rule
        super().__init__(
            coin=coin,
            init_fortune=init_fortune,
            seed=seed,
        )

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, new_p: float):
        self._p = new_p

    @property
    def rule(self) -> Callable:
        return self._rule

    @rule.setter
    def rule(self, new_rule: Callable):
        self._rule = new_rule

    def strat(self, outcomes: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(data=outcomes, columns=['outcomes'])

        # Which side is biased?
        which_side = self.p >= 0.5

        sizes = np.empty(len(outcomes))

        # Custom fractional bet
        fraction = self.rule(self.p) if which_side else self.rule(1 - self.p)

        fortune = self.init_fortune

        for t, outcome in enumerate(outcomes):
            sizes[t] = fraction * fortune
            fortune *= 1 + fraction if which_side == outcome else 1 - fraction

        df['bet'] = which_side
        df['size'] = sizes

        return df


class EmpiricalArbitraryBet(Bet):
    """Same as ExactArbitrary but the parameter p is estimated on-the-fly
    rather than known in advance, much like in a bandit context.
    """
    def __init__(
        self,
        coin: Type[Coin],
        init_fortune: float = 100.0,
        rule: Callable = None,
        warmup_time: int = 1,
        seed: float = 0,
    ):
        self._rule = rule
        self._warmup_time = warmup_time
        super().__init__(
            coin=coin,
            init_fortune=init_fortune,
            seed=seed,
        )
        
    @property
    def rule(self) -> Callable:
        return self._rule

    @rule.setter
    def rule(self, new_rule: Callable):
        self._rule = new_rule

    @property
    def warmup_time(self) -> int:
        return self._warmup_time

    @warmup_time.setter
    def warmup_time(self, new_warmup_time: int):
        self._warmup_time = new_warmup_time

    def strat(self, outcomes: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(data=outcomes, columns=['outcomes'])

        sizes = np.empty(len(outcomes))

        fortune = self.init_fortune

        for t, outcome in enumerate(outcomes):
            if t < self.warmup_time:
                p = 0.5
            else:
                p = outcomes.iloc[:t].mean()
            # Which side seems biased?
            which_side = p >= 0.5
            # Custom fractional bet
            fraction = self.rule(p) if which_side else self.rule(1 - p)

            sizes[t] = fraction * fortune
            fortune *= 1 + fraction if which_side == outcome else 1 - fraction

        df['bet'] = which_side
        df['size'] = sizes

        return df


class ExactBellmanBet(ExactArbitraryBet):
    """Bet sizing by maximization of a log utility with known probability p.
    If p is indeed the probability to win a single occurrence, this strategy is
    optimal (in terms of log-utility by definition), but if p is mispecified
    (for example if a coin toss wins with 60% probability and the Bellman
    criterion is used with p=80%) this strategy can fail and lead to
    huge losses.
    """
    def __init__(
        self,
        coin: Type[Coin],
        p: float = 0.5,
        init_fortune: float = 100.0,
        seed: float = 0,
    ):
        super().__init__(
            coin=coin,
            p=p,
            init_fortune=init_fortune,
            rule=lambda p: 2 * p - 1,
            seed=seed,
        )


class EmpiricalBellmanBet(EmpiricalArbitraryBet):
    """Same as exact Bellman but the parameter p is estimated on-the-fly
    rather than known in advance, much like in a bandit context.
    """
    def __init__(
        self,
        coin: Type[Coin],
        init_fortune: float = 100.0,
        warmup_time: int = 1,
        seed: float = 0,
    ):
        super().__init__(
            coin=coin,
            init_fortune=init_fortune,
            rule=lambda p: 2 * p - 1,
            warmup_time=warmup_time,
            seed=seed,
        )
