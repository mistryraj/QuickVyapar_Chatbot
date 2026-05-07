from typing import Optional, List, Literal
from pydantic import BaseModel, Field

Intent = Literal["PRODUCT_QUERY", "NEGOTIATE", "REQUEST_HUMAN", "OFF_TOPIC", "GREETING", "FAREWELL"]


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ProductLite(BaseModel):
    post_id: str
    title: str
    price: int
    priceUnitType: str
    categoryName: str
    image: Optional[str] = None
    user_name: Optional[str] = None


class NegotiationResult(BaseModel):
    decision: Literal["accept", "counter", "reject"]
    listed_price: int
    buyer_offer: Optional[int]
    counter_price: Optional[int]
    min_price: int
    round_num: int
    product_id: str


class ChatResponse(BaseModel):
    reply: str
    intent: Intent
    products: List[ProductLite] = []
    negotiation: Optional[NegotiationResult] = None
    notify_seller: bool = False
    end_chat: bool = False
