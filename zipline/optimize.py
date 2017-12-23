"""
参考cvxportfolio

尚未完成
"""

import cvxpy as cvx

class Objective():
    """Base class for objectives."""
    pass


class TargetWeights(weights):
    pass
#------------------------------------------------------------------------------
#                                Constraints                                  #
#------------------------------------------------------------------------------
class Constraint():
    """
    Base class for constraints.
    """
    pass


class MaxGrossExposure():
    """
    Constraint on the maximum gross exposure for the portfolio.

    Requires that the sum of the absolute values of the portfolio
    weights be less than `max`.
    """
    def __init__(self, max_value):
        """
        Parameters
        ----------
        max_value (float) -- The maximum gross exposure of the portfolio.

        Examples
        --------
        `MaxGrossExposure(1.5)` constrains the total value of the portfolios
        longs and shorts to be no more than 1.5x the current portfolio value.
        """
        return super().__init__(**kwargs)


class NetExposure():
    """
    Constraint on the net exposure of the portfolio.

    Requires that the sum of the weights (positive or negative) of all assets
    in the portfolio fall between `min` and `max`.
    """
    def __init__(self, min_value, max_value):
        """
        Parameters
        ----------
        min_value : float
            The minimum net exposure of the portfolio.

        max_value : float
            The maximum net exposure of the portfolio.

        Examples
        --------
        `NetExposure(-0.1, 0.1)` constrains the difference in value between 
        the portfolio's longs and shorts to be between -10% and 10% of the 
        current portfolio value.
        """
        return super().__init__(**kwargs)


class DollarNeutral():
    """
    Constraint requiring that long and short exposures must be equal-sized.
    限制多空总头寸风险敞口等规模。
    Requires that the sum of the weights (positive or negative) of all assets
    in the portfolio fall between +-(tolerance).
    """
    def __init__(self, tolerance=0.0001):
        """
        Parameters
        ----------
        tolerance : float, optional
            Allowed magnitude of net market exposure. Default is 0.0001

        """
        return super().__init__(**kwargs)


class NetGroupExposure():
    """
    Constraint requiring bounded net exposure to groups of assets.

    Groups are defined by map from (asset -> label). Each unique label 
    generates a constraint specifying that the sum of the weights of
    assets mapped to that label should fall between a lower and upper bounds.

    Min/Max group exposures are specified as maps from (label -> float).
    """
    def __init__(self, labels, min_weights, max_weights, etf_lookthru=None):
        """
        Parameters
        ----------
        labels : pd.Series[Asset -> object] or dict[Asset -> object]
            Map from asset -> group label

        min_weights  : pd.Series[Asset -> object] or dict[Asset -> object]
            Map from group label to minimum net exposure to assets in that group.

        max_weights : pd.Series[Asset -> object] or dict[Asset -> object]
            Map from group label to maximum net exposure to assets in that group.

        etf_lookthru : pd.DataFrame, optional
            未完成
            Indexed by constituent assets x ETFs, expresses the weight of each 
            constituent in each ETF.

            A DataFrame containing ETF constituents data. Each column of the frame 
            should contain weights (from 0.0 to 1.0) representing the holdings for
            an ETF. Each row should contain weights for a single stock in each ETF. 
            Columns should sum approximately to 1.0. If supplied, ETF holdings in 
            the current and target portfolio will be decomposed into their constituents 
            before constraints are applied.

        """
        return super().__init__(**kwargs)

    @classmethod
    def with_equal_bounds(labels, min_value, max_value, etf_lookthru=None):
        """
        Special case constructor that applies static lower and upper bounds to all groups.

        Parameters
        ----------
        labels : pd.Series[Asset -> object] or dict[Asset -> object]
           Map from asset -> group label.
        min_value : float
            Lower bound for exposure to any group.
        max_value  : float
            Upper bound for exposure to any group.
        """
        pass


class PositionConcentration():
    """
    Constraint enforcing minimum/maximum position weights.
    """
    def __init__(self, min_weights, max_weights, default_min_weight=0.0, 
                 default_max_weight=0.0, etf_lookthru=None):
        """
        Parameters
        ----------
        min_weights : pd.Series[Asset -> float] or dict[Asset -> float]
            Map from asset to minimum position weight for that asset.

        max_weights  : pd.Series[Asset -> float] or dict[Asset -> float]
            Map from asset to maximum position weight for that asset.

        default_min_weight : float, optional
            Value to use as a lower bound for assets not found in min_weights. 
            Default is 0.0.

        default_max_weight : float, optional
            Value to use as a lower bound for assets not found in max_weights. 
            Default is 0.0.

        etf_lookthru : pd.DataFrame, optional
            Indexed by constituent assets x ETFs, expresses the weight of each 
            constituent in each ETF.
            A DataFrame containing ETF constituents data. Each column of the frame 
            should contain weights (from 0.0 to 1.0) representing the holdings for 
            an ETF. Each row should contain weights for a single stock in each ETF. 
            
            Columns should sum approximately to 1.0. If supplied, ETF holdings in 
            the current and target portfolio will be decomposed into their constituents 
            before constraints are applied.

        """
        return super().__init__(**kwargs)

    @classmethod
    def with_equal_bounds(min_value, max_value, etf_lookthru=None):
        """
        Special case constructor that applies static lower and upper bounds to all groups.

        Parameters
        ----------
        min_value : float
            Minimum position weight for all assets.
        max_value  : float
            Maximum position weight for all assets.
        """
        pass


class FactorExposure():
    """
    Constraint requiring bounded net exposure to a set of risk factors.

    Factor loadings are specified as a DataFrame of floats whose columns 
    are factor labels and whose index contains Assets. Minimum and maximum 
    factor exposures are specified as maps from factor label to min/max net 
    exposure.

    For each column in the loadings frame, we constrain:
    `(new_weights * loadings[column]).sum() >= min_exposure[column]`
    `(new_weights * loadings[column]).sum() <= max_exposure[column]`
    """
    def __init__(self, loadings, min_exposures, max_exposures):
        """
        Parameters
        ----------
        loadings : pd.DataFrame
            An (assets x labels) frame of weights for each (asset, factor) pair.
        min_exposures : dict or pd.Series
            Minimum net exposure values for each factor.
        max_exposures : dict or pd.Series
            Maximum net exposure values for each factor.
        """
        return super().__init__(**kwargs)


class Pair():
    """
    A constraint representing a pair of inverse-weighted stocks.

    """
    def __init__(self, long, short, hedge_ratio=1.0, tolerance=0.0):
        """
        Parameters
        ----------
        long : Asset
            The asset to long.
        short : Asset
            The asset to short.
        hedge_ratio : float, optional
            The ratio between the respective absolute values of the long 
            and short weights. Required to be greater than 0. Default is 
            1.0, signifying equal weighting.
        olerance : float, optional
            The amount by which the hedge ratio of the calculated weights 
            is allowed to differ from the given hedge ratio, in either 
            direction. Required to be greater than or equal to 0. Default 
            is 0.0.
        """
        return super().__init__(**kwargs)


class Basket():
    """
    Constraint requiring bounded net exposure to a basket of stocks.

    """
    def __init__(self, assets, min_net_exposure, max_net_exposure):
        """
        Parameters
        ----------
        assets : iterable[Asset]
            Assets to be constrained.
        min_net_exposure : float
            Minimum allowed net exposure to the basket.
        max_net_exposure : float
            Maximum allowed net exposure to the basket.
        """
        return super().__init__(**kwargs)


class Frozen():
    """
    Constraint for assets whose positions cannot change.

    """
    def __init__(self, asset_or_assets, max_error_display=10):
        """
        Parameters
        ----------
        asset_or_assets : Asset or sequence[Asset]
            Asset(s) whose weight(s) cannot change.
        """
        return super().__init__(**kwargs)


class ReduceOnly():
    """
    Constraint for assets whose weights can only move toward zero 
    and cannot cross zero.

    """
    def __init__(self, asset_or_assets, max_error_display=10):
        """
        Parameters
        ----------
        asset : Asset
            The asset whose position weight cannot increase in magnitude.
        """
        return super().__init__(**kwargs)


class LongOnly():
    """
    Constraint for assets that cannot be held in short positions.

    """
    def __init__(self, asset_or_assets, max_error_display=10):
        """
        Parameters
        ----------
        asset_or_assets : Asset or iterable[Asset]  
            The asset(s) that must be long or zero.
        """
        return super().__init__(**kwargs)


class ShortOnly():
    """
    Constraint for assets that cannot be held in long positions.

    """
    def __init__(self, asset_or_assets, max_error_display=10):
        """
        Parameters
        ----------
        asset_or_assets : Asset or iterable[Asset]  
            The asset(s) that must be short or zero.
        """
        return super().__init__(**kwargs)


class FixedWeight():
    """
    A constraint representing an asset whose position weight is fixed
    as a specified value.

    """
    def __init__(self, asset, weight):
        """
        Parameters
        ----------
        asset : Asset
            The asset whose weight is fixed.
        weight : float
            The weight at which asset should be fixed.
        """
        return super().__init__(**kwargs)


class CannotHold():
    """
    Constraint for assets whose position sizes must be 0.

    """
    def __init__(self, asset, weight):
        """
        Parameters
        ----------
        asset_or_assets : Asset or iterable[Asset]
            The asset(s) that cannot be held.
        """
        return super().__init__(**kwargs)


#------------------------------------------------------------------------------
#                              Results and Errors                             #
#------------------------------------------------------------------------------


class InfeasibleConstraints(ValueError):
    """
    Raised when an optimization fails because there are no valid portfolios.

    This most commonly happens when the weight in some asset is simultaneously
    constrained to be above and below some threshold.

    不可行限制。如 `x > 0` and `x < 0`
    """
    pass


class UnboundedObjective(ValueError):
    """
    Raised when an optimization fails because at least one weight in the
   'optimal' portfolio is 'infinity'.

    More formally, raised when an optimization fails because the value of 
    an objective function improves as a value being optimized grows toward 
    infinity, and no constraint puts a bound on the magnitude of that value.

    无限制边界，如 x 无限制，其最大值 'infinity'
    """
    pass


class OptimizationFailed(ValueError):
    """
    Generic exception raised when an optimization fails a reason with no 
    special metadata.
    """
    pass


class OptimizationResult():
    """
    The result of an optimization.
    """
    def raise_for_status():
        """
        Raise an error if the optimization did not succeed.

        """
        pass

    def print_diagnostics():
        """
        Print diagnostic information gathered during the optimization.

        """
        pass

    @property
    def old_weights(self):
        """
        pandas.Series -- Portfolio weights before the optimization.
        """
        pass

    @property
    def new_weights(self):
        """
        pandas.Series or None -- New optimal weights, or None if the optimization failed.
        """
        pass

    @property
    def diagnostics(self):
        """
        diagnostics -- 包含有关违反（或可能违反）约束的诊断信息的对象。
        """
        pass

    @property
    def status(self):
        """
        str -- String indicating the status of the optimization.
        """
        pass

    @property
    def success(self):
        """
        class:bool -- True if the optimization successfully produced a result.
        """
        pass


def order_optimal_portfolio(objective, constraints):
    """
    Calculate an optimal portfolio and place orders toward that portfolio.

    Parameters
    ----------
    objective : Objective
        The objective to be minimized/maximized by the new portfolio.
    constraints :list[Constraint]
        Constraints that must be respected by the new portfolio.

    Raises
    ------
    InfeasibleConstraints : InfeasibleConstraints
        Raised when there is no possible portfolio that satisfies the received constraints.
    UnboundedObjective : UnboundedObjective
        Raised when the received constraints are not sufficient to put an upper (or lower) 
        bound on the calculated portfolio weights.

    Returns
    -------
    order_ids  : pd.Series[Asset -> str]
        The unique identifiers for the orders that were placed.
    """
    pass


def calculate_optimal_portfolio(objective, constraints, current_portfolio=None):
    """
    Calculate optimal portfolio weights given `objective` and `constraints`.

    Parameters
    ----------
    objective : Objective
        The objective to be minimized/maximized by the new portfolio.
    constraints :list[Constraint]
        Constraints that must be respected by the new portfolio.
    current_portfolio:pd.Series, optional
        A Series containing the current portfolio weights, expressed as percentages
        of the portfolio's liquidation value.
        When called from a trading algorithm, the default value of current_portfolio
        is the algorithm's current portfolio.
        When called interactively, the default value of current_portfolio is an empty
        portfolio.

    Raises
    ------
    InfeasibleConstraints : InfeasibleConstraints
        Raised when there is no possible portfolio that satisfies the received constraints.
    UnboundedObjective : UnboundedObjective
        Raised when the received constraints are not sufficient to put an upper (or lower) 
        bound on the calculated portfolio weights.

    Returns
    -------
    optimal_portfolio : pd.Series
        A Series containing portfolio weights that maximize (or minimize) objective without
        violating any constraints. Weights should be interpreted in the same way as 
        current_portfolio
    """
    pass


def run_optimization(objective, constraints, current_portfolio=None):
    """
    Run a portfolio optimization.

    Parameters
    ----------
    objective : Objective
        The objective to be minimized/maximized

    constraints :list[Constraint]
        List of constraints that must be satisfied by the new portfolio

    current_portfolio:pd.Series, optional
        A Series containing the current portfolio weights, expressed as
        percentages of the portfolio's liquidation value.

        When called from a trading algorithm, the default value of 
        `current_portfolio` is the algorithm's current portfolio.

        When called interactively, the default value of `current_portfolio`
        is an empty portfolio.

    Returns
    -------
    result  : optimize.OptimizationResult
        An object containing information about the result of the optimization.

    """
    pass