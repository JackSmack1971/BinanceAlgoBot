from __future__ import annotations

from typing import Literal, TypedDict

from ..trading.types import Price, Quantity, Symbol

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]
TimeInForce = Literal["GTC", "IOC", "FOK"]


class OrderRequest(TypedDict):
    symbol: Symbol
    side: Side
    quantity: Quantity
    order_type: OrderType
    price: Price | None
    time_in_force: TimeInForce


class OrderResponse(TypedDict, total=False):
    order_id: int
    symbol: Symbol
    status: str
