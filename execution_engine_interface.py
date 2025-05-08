# execution_engine_interface.py
from abc import ABC, abstractmethod
import pandas as pd

class ExecutionEngineInterface(ABC):
    """
    Interface for the execution engine.
    """

    @abstractmethod
    async def execute_trades(self, signals: pd.DataFrame):
        """
        Execute trades based on signals from the strategy.

        Args:
            signals (pd.DataFrame): DataFrame containing signals.
        """
        pass