from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm4ad.method.traceaad.portfolio import (
    OperatorPortfolio,
    PortfolioWeights,
    aggregate_batch_utility,
    signed_utility,
)
from llm4ad.method.traceaad.schema import OperatorName


@dataclass
class _StubOperator:
    name: str
    role: str = "exploit"
    _eligible: bool = True

    def trigger(self, ctx) -> bool:
        return self._eligible


def test_signed_utility_is_bounded_and_keeps_sign() -> None:
    assert signed_utility(0.0) == pytest.approx(0.0)
    assert signed_utility(10.0) < 1.0
    assert signed_utility(-10.0) > -1.0
    assert signed_utility(-0.2) < signed_utility(0.0) < signed_utility(0.2)


def test_batch_utility_is_max_plus_mean() -> None:
    utilities = [0.2, -0.4, 0.8]
    assert aggregate_batch_utility(utilities) == pytest.approx(0.5 * 0.8 + 0.5 * (0.2 - 0.4 + 0.8) / 3)
    assert aggregate_batch_utility([]) == pytest.approx(-1.0)


def test_portfolio_records_one_batch_outcome_per_operator_iteration() -> None:
    op = _StubOperator("test_operator")
    portfolio = OperatorPortfolio((op,), PortfolioWeights())

    portfolio.update_batch(
        op=op,
        iteration=7,
        batch_utility=0.4,
        downside=0.0,
        best_valid=True,
        best_novel=False,
        total_cost=123.0,
        global_best=True,
    )

    stats = portfolio.snapshot()[op.name]
    assert stats["attempt_count"] == 1
    assert stats["mean_utility"] == pytest.approx(0.4)
    assert stats["valid_rate"] == pytest.approx(1.0)
    assert stats["novel_rate"] == pytest.approx(0.0)
    assert stats["mean_downside"] == pytest.approx(0.0)
    assert stats["mean_cost"] == pytest.approx(123.0)
    assert stats["global_best_rate"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="already updated"):
        portfolio.update_batch(
            op=op,
            iteration=7,
            batch_utility=-1.0,
            downside=1.0,
            best_valid=False,
            best_novel=False,
            total_cost=999.0,
        )


def test_portfolio_credit_follows_recent_outcomes_instead_of_lifetime_mean() -> None:
    op = _StubOperator("test_operator")
    portfolio = OperatorPortfolio((op,), PortfolioWeights())

    for iteration in range(8):
        portfolio.update_batch(
            op=op,
            iteration=iteration,
            batch_utility=-1.0,
            downside=1.0,
            best_valid=True,
            best_novel=False,
            total_cost=10.0,
        )
    for iteration in range(8, 16):
        portfolio.update_batch(
            op=op,
            iteration=iteration,
            batch_utility=1.0,
            downside=0.0,
            best_valid=True,
            best_novel=True,
            total_cost=10.0,
        )

    assert portfolio.snapshot()[op.name]["mean_utility"] > 0.5


def test_global_best_bonus_increases_an_operators_selection_probability() -> None:
    record_op = _StubOperator("record_operator")
    ordinary_op = _StubOperator("ordinary_operator")
    weights = PortfolioWeights(
        alpha=0.0,
        lambda_downside=0.0,
        lambda_cost=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=1.0,
        near_record_bonus=0.0,
        ucb_c=0.0,
        epsilon_init=0.0,
        epsilon_end=0.0,
        min_probability=0.0,
    )
    portfolio = OperatorPortfolio((record_op, ordinary_op), weights)
    common = dict(
        iteration=1,
        batch_utility=0.0,
        downside=0.0,
        best_valid=True,
        best_novel=False,
        total_cost=10.0,
    )
    portfolio.update_batch(op=record_op, global_best=True, **common)
    portfolio.update_batch(op=ordinary_op, global_best=False, **common)

    probabilities = portfolio.probabilities(ctx=None, iteration=1, max_iter=10)

    assert probabilities[record_op.name] > probabilities[ordinary_op.name]


def test_near_record_bonus_increases_operator_probability() -> None:
    near = _StubOperator("near")
    ordinary = _StubOperator("ordinary")
    weights = PortfolioWeights(
        alpha=0.0,
        lambda_downside=0.0,
        lambda_cost=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=0.0,
        near_record_bonus=1.0,
        ucb_c=0.0,
        epsilon_init=0.0,
        epsilon_end=0.0,
        min_probability=0.0,
    )
    portfolio = OperatorPortfolio((near, ordinary), weights)
    common = dict(
        iteration=1,
        batch_utility=0.0,
        downside=0.0,
        best_valid=True,
        best_novel=False,
        total_cost=0.0,
        global_best=False,
    )
    portfolio.update_batch(op=near, near_record=True, **common)
    portfolio.update_batch(op=ordinary, near_record=False, **common)

    probabilities = portfolio.probabilities(ctx=None, iteration=1, max_iter=10)

    assert probabilities[near.name] > probabilities[ordinary.name]


def test_late_phase_caps_novelty_without_removing_softmax_exploration() -> None:
    novelty = _StubOperator(OperatorName.NOVELTY)
    exploit = _StubOperator(OperatorName.ENDPOINT)
    weights = PortfolioWeights(
        alpha=10.0,
        lambda_downside=0.0,
        lambda_cost=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=0.0,
        ucb_c=0.0,
        epsilon_init=0.0,
        epsilon_end=0.0,
        min_probability=0.05,
        late_novelty_max_probability=0.2,
    )
    portfolio = OperatorPortfolio((novelty, exploit), weights)
    common = dict(
        iteration=0,
        downside=0.0,
        best_valid=True,
        best_novel=False,
        total_cost=0.0,
    )
    portfolio.update_batch(op=novelty, batch_utility=1.0, **common)
    portfolio.update_batch(op=exploit, batch_utility=-1.0, **common)

    early = portfolio.probabilities(ctx=None, iteration=0, max_iter=100)
    late = portfolio.probabilities(ctx=None, iteration=90, max_iter=100)

    assert min(early.values()) >= 0.05
    assert late[OperatorName.NOVELTY] == pytest.approx(0.2)


def test_global_exploration_mixture_keeps_all_eligible_positive() -> None:
    a = _StubOperator("a")
    b = _StubOperator("b")
    weights = PortfolioWeights(
        alpha=5.0,
        lambda_downside=0.0,
        lambda_cost=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=0.0,
        ucb_c=0.0,
        epsilon_init=0.2,
        epsilon_end=0.2,
        min_probability=0.0,
    )
    portfolio = OperatorPortfolio((a, b), weights)
    common = dict(iteration=0, downside=0.0, best_valid=True, best_novel=False, total_cost=0.0)
    portfolio.update_batch(op=a, batch_utility=1.0, **common)
    portfolio.update_batch(op=b, batch_utility=-1.0, iteration=1, **{k: v for k, v in common.items() if k != "iteration"})

    probabilities = portfolio.probabilities(ctx=None, iteration=0, max_iter=100)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-9
    assert min(probabilities.values()) >= 0.2 / 2 - 1e-9


def test_new_weight_names_are_not_overridden_by_legacy_defaults() -> None:
    op = _StubOperator("op")
    portfolio = OperatorPortfolio(
        (op,),
        PortfolioWeights(lambda_downside=0.0, lambda_cost=0.0),
    )
    portfolio.update_batch(
        op=op,
        iteration=0,
        batch_utility=0.0,
        downside=1.0,
        best_valid=True,
        best_novel=False,
        total_cost=0.0,
    )

    components = portfolio.evaluate(
        ctx=None, iteration=0, max_iter=10
    ).components[op.name]
    assert components["score"] == pytest.approx(components["prior"] + components["ucb"])


def test_discounted_mass_drives_ucb_and_prior_fades() -> None:
    op = _StubOperator("op")
    portfolio = OperatorPortfolio(
        (op,),
        PortfolioWeights(ema_decay=0.5, ucb_c=1.0, prior_pseudo_count=2.0),
    )
    for iteration in range(5):
        portfolio.mark_eligible([op.name])
        portfolio.update_batch(
            op=op,
            iteration=iteration,
            batch_utility=0.0,
            downside=0.0,
            best_valid=True,
            best_novel=False,
            total_cost=0.0,
        )

    stats = portfolio.snapshot()[op.name]
    assert stats["discounted_mass"] < stats["attempt_count"]
    first = portfolio.evaluate(ctx=None, iteration=5, max_iter=10)
    first_prior = first.components[op.name]["prior"]
    portfolio.mark_eligible([op.name])
    portfolio.update_batch(
        op=op,
        iteration=5,
        batch_utility=0.0,
        downside=0.0,
        best_valid=True,
        best_novel=False,
        total_cost=0.0,
    )
    second = portfolio.evaluate(ctx=None, iteration=6, max_iter=10)
    assert second.components[op.name]["prior"] < first_prior


def test_stagnation_increases_global_exploration_mass() -> None:
    a = _StubOperator("a")
    b = _StubOperator("b")
    portfolio = OperatorPortfolio(
        (a, b),
        PortfolioWeights(
            alpha=5.0,
            epsilon_init=0.05,
            epsilon_end=0.05,
            prior_mean=0.0,
            ucb_c=0.0,
            min_probability=0.0,
        ),
    )
    portfolio.update_batch(
        op=a,
        iteration=0,
        batch_utility=1.0,
        downside=0.0,
        best_valid=True,
        best_novel=False,
        total_cost=0.0,
    )
    portfolio.update_batch(
        op=b,
        iteration=1,
        batch_utility=-1.0,
        downside=1.0,
        best_valid=True,
        best_novel=False,
        total_cost=0.0,
    )
    early = portfolio.probabilities(ctx=None, iteration=0, max_iter=100, stagnation=0)
    stalled = portfolio.probabilities(
        ctx=None, iteration=0, max_iter=100, stagnation=12
    )
    assert stalled[a.name] < early[a.name]
    assert portfolio._epsilon(0, 100, 12) > portfolio._epsilon(0, 100, 0)


def test_new_eligible_operator_does_not_get_infinite_ucb() -> None:
    always = _StubOperator("always")
    rare = _StubOperator("rare", _eligible=False)
    weights = PortfolioWeights(ucb_c=1.0, ucb_n0=1.0, epsilon_init=0.0, epsilon_end=0.0, min_probability=0.0)
    portfolio = OperatorPortfolio((always, rare), weights)
    for i in range(20):
        portfolio.mark_eligible(["always"])
        portfolio.update_batch(
            op=always,
            iteration=i,
            batch_utility=0.1,
            downside=0.0,
            best_valid=True,
            best_novel=False,
            total_cost=1.0,
        )
    rare._eligible = True
    decision = portfolio.evaluate(ctx=None, iteration=20, max_iter=100, mark_eligibility=True)
    assert decision.components["rare"]["ucb"] < 2.0
    assert decision.probabilities["rare"] < 1.0


def test_ineligible_operators_are_absent_from_probabilities() -> None:
    yes = _StubOperator("yes")
    no = _StubOperator("no", _eligible=False)
    portfolio = OperatorPortfolio((yes, no), PortfolioWeights())
    probabilities = portfolio.probabilities(ctx=object(), iteration=0, max_iter=10)
    assert list(probabilities) == ["yes"]
    assert probabilities["yes"] == pytest.approx(1.0)


def test_temperature_floor_prevents_late_softmax_from_freezing() -> None:
    better = _StubOperator("better")
    worse = _StubOperator("worse")
    weights = PortfolioWeights(
        alpha=1.0,
        lambda_downside=0.0,
        lambda_cost=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=0.0,
        ucb_c=0.0,
        temperature_init=0.0,
        temperature_end=0.0,
        temperature_floor=0.5,
        epsilon_init=0.0,
        epsilon_end=0.0,
        min_probability=0.0,
    )
    portfolio = OperatorPortfolio((better, worse), weights)
    common = dict(
        iteration=0,
        downside=0.0,
        best_valid=True,
        best_novel=False,
        total_cost=0.0,
    )
    portfolio.update_batch(op=better, batch_utility=1.0, **common)
    portfolio.update_batch(op=worse, batch_utility=-1.0, **common)

    probabilities = portfolio.probabilities(ctx=None, iteration=100, max_iter=100)

    assert probabilities[worse.name] > 0.01


def test_actual_batch_cost_remains_comparable_in_portfolio_selection() -> None:
    fast = _StubOperator("fast")
    slow = _StubOperator("slow")
    weights = PortfolioWeights(
        alpha=0.0,
        lambda_downside=0.0,
        lambda_cost=1.0,
        delta_r=0.0,
        delta_c=1.0,
        global_best_bonus=0.0,
        ucb_c=0.0,
        epsilon_init=0.0,
        epsilon_end=0.0,
        min_probability=0.0,
        cost_scale=120.0,
    )
    portfolio = OperatorPortfolio((fast, slow), weights)
    common = dict(
        iteration=0,
        batch_utility=0.0,
        downside=0.0,
        best_valid=True,
        best_novel=False,
    )
    portfolio.update_batch(op=fast, total_cost=30.0, **common)
    portfolio.update_batch(op=slow, total_cost=300.0, **common)

    probabilities = portfolio.probabilities(ctx=None, iteration=0, max_iter=100)

    assert probabilities[fast.name] > probabilities[slow.name]
