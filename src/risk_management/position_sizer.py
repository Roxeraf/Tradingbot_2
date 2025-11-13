"""
Position sizing module for risk management
Calculates appropriate position sizes based on risk parameters
"""
from typing import Optional
from enum import Enum


class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED_PERCENTAGE = "fixed_percentage"
    RISK_BASED = "risk_based"
    KELLY_CRITERION = "kelly_criterion"
    FIXED_AMOUNT = "fixed_amount"


class PositionSizer:
    """
    Calculate position sizes based on risk management rules
    """

    def __init__(
        self,
        max_position_size: float = 0.1,
        max_risk_per_trade: float = 0.02,
        max_portfolio_risk: float = 0.06,
        sizing_method: PositionSizingMethod = PositionSizingMethod.RISK_BASED
    ):
        """
        Initialize position sizer

        Args:
            max_position_size: Maximum position size as % of portfolio (e.g., 0.1 = 10%)
            max_risk_per_trade: Maximum risk per trade as % of portfolio (e.g., 0.02 = 2%)
            max_portfolio_risk: Maximum total portfolio risk exposure (e.g., 0.06 = 6%)
            sizing_method: Position sizing method to use
        """
        self.max_position_size = max_position_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.sizing_method = sizing_method

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        confidence: float = 1.0,
        current_exposure: float = 0.0
    ) -> float:
        """
        Calculate position size using the configured sizing method

        Args:
            account_balance: Total account balance
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            confidence: Signal confidence (0-1), used to scale position size
            current_exposure: Current portfolio risk exposure (0-1)

        Returns:
            Position size in base currency (e.g., amount of BTC to buy)
        """
        if self.sizing_method == PositionSizingMethod.FIXED_PERCENTAGE:
            return self._fixed_percentage_sizing(account_balance, entry_price, confidence)

        elif self.sizing_method == PositionSizingMethod.RISK_BASED:
            return self._risk_based_sizing(
                account_balance,
                entry_price,
                stop_loss_price,
                confidence,
                current_exposure
            )

        elif self.sizing_method == PositionSizingMethod.FIXED_AMOUNT:
            return self._fixed_amount_sizing(entry_price)

        else:
            # Default to risk-based
            return self._risk_based_sizing(
                account_balance,
                entry_price,
                stop_loss_price,
                confidence,
                current_exposure
            )

    def _fixed_percentage_sizing(
        self,
        account_balance: float,
        entry_price: float,
        confidence: float
    ) -> float:
        """
        Calculate position size as fixed percentage of portfolio

        Args:
            account_balance: Total account balance
            entry_price: Entry price
            confidence: Signal confidence

        Returns:
            Position size in base currency
        """
        position_value = account_balance * self.max_position_size * confidence
        position_size = position_value / entry_price
        return position_size

    def _risk_based_sizing(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        confidence: float,
        current_exposure: float
    ) -> float:
        """
        Calculate position size based on risk per trade

        Args:
            account_balance: Total account balance
            entry_price: Entry price
            stop_loss_price: Stop loss price
            confidence: Signal confidence
            current_exposure: Current portfolio risk exposure

        Returns:
            Position size in base currency
        """
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)

        if risk_per_unit == 0:
            # If no stop loss defined, use fixed percentage
            return self._fixed_percentage_sizing(account_balance, entry_price, confidence)

        # Check if adding this trade would exceed maximum portfolio risk
        remaining_risk_capacity = self.max_portfolio_risk - current_exposure

        if remaining_risk_capacity <= 0:
            # Already at maximum risk, don't take new position
            return 0

        # Calculate risk amount (use the lesser of max_risk_per_trade and remaining capacity)
        risk_percentage = min(self.max_risk_per_trade, remaining_risk_capacity)
        risk_amount = account_balance * risk_percentage * confidence

        # Calculate position size based on risk
        position_size = risk_amount / risk_per_unit

        # Apply maximum position size limit
        max_position_value = account_balance * self.max_position_size
        max_position_size = max_position_value / entry_price

        position_size = min(position_size, max_position_size)

        return position_size

    def _fixed_amount_sizing(self, entry_price: float, fixed_amount: float = 100.0) -> float:
        """
        Calculate position size based on fixed currency amount

        Args:
            entry_price: Entry price
            fixed_amount: Fixed amount to invest

        Returns:
            Position size in base currency
        """
        return fixed_amount / entry_price

    def _kelly_criterion_sizing(
        self,
        account_balance: float,
        entry_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float
    ) -> float:
        """
        Calculate position size using Kelly Criterion

        Args:
            account_balance: Total account balance
            entry_price: Entry price
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount (as ratio, e.g., 1.5 = 50% gain)
            avg_loss: Average loss amount (as ratio, e.g., 0.9 = 10% loss)
            confidence: Signal confidence

        Returns:
            Position size in base currency
        """
        if avg_loss >= 1.0 or avg_win <= 1.0:
            # Invalid parameters
            return 0

        # Kelly formula: f = (p * b - q) / b
        # where:
        # p = win probability
        # q = loss probability (1 - p)
        # b = win/loss ratio
        p = win_rate
        q = 1 - win_rate
        b = abs(avg_win - 1) / abs(1 - avg_loss)

        kelly_fraction = (p * b - q) / b

        # Apply fractional Kelly (typically 0.25 to 0.5 of full Kelly for safety)
        kelly_fraction = max(0, kelly_fraction * 0.25)

        # Scale by confidence
        kelly_fraction *= confidence

        # Apply maximum position size limit
        position_fraction = min(kelly_fraction, self.max_position_size)

        position_value = account_balance * position_fraction
        position_size = position_value / entry_price

        return position_size

    def calculate_risk_amount(
        self,
        position_size: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate the amount at risk for a position

        Args:
            position_size: Position size in base currency
            entry_price: Entry price
            stop_loss_price: Stop loss price

        Returns:
            Amount at risk in quote currency
        """
        risk_per_unit = abs(entry_price - stop_loss_price)
        risk_amount = position_size * risk_per_unit
        return risk_amount

    def calculate_risk_percentage(
        self,
        position_size: float,
        entry_price: float,
        stop_loss_price: float,
        account_balance: float
    ) -> float:
        """
        Calculate risk as percentage of account balance

        Args:
            position_size: Position size in base currency
            entry_price: Entry price
            stop_loss_price: Stop loss price
            account_balance: Total account balance

        Returns:
            Risk percentage (0-1)
        """
        risk_amount = self.calculate_risk_amount(position_size, entry_price, stop_loss_price)

        if account_balance == 0:
            return 0

        return risk_amount / account_balance

    def validate_position_size(
        self,
        position_size: float,
        entry_price: float,
        account_balance: float,
        stop_loss_price: Optional[float] = None
    ) -> bool:
        """
        Validate if position size adheres to risk management rules

        Args:
            position_size: Proposed position size
            entry_price: Entry price
            account_balance: Account balance
            stop_loss_price: Stop loss price (optional)

        Returns:
            True if position size is valid
        """
        # Check position value doesn't exceed max position size
        position_value = position_size * entry_price
        max_position_value = account_balance * self.max_position_size

        if position_value > max_position_value:
            return False

        # If stop loss provided, check risk doesn't exceed max risk per trade
        if stop_loss_price:
            risk_percentage = self.calculate_risk_percentage(
                position_size,
                entry_price,
                stop_loss_price,
                account_balance
            )

            if risk_percentage > self.max_risk_per_trade:
                return False

        return True
