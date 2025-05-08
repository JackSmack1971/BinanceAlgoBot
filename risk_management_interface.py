# risk_management_interface.py
from abc import ABC, abstractmethod

class RiskManagementInterface(ABC):
    """
    Interface for the risk management component.
    """

    @abstractmethod
    def manage_risk(self):
        """
        Manage risk for the trading strategy.
        """
        pass