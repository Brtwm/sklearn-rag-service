from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    """Question coming from the user"""
    question: str = Field(..., min_length=1, max_length=500)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, question: str) -> str:
        if not question.strip():
            raise ValueError("Question must contain text")
        return question


class Source(BaseModel):
    """One retrieved chunk that supported the answer"""
    url: str
    snippet: str = Field(..., description="First ~200 chars of the chunk")


class ChatResponse(BaseModel):
    """Final answer with citations"""
    answer: str
    sources: list[Source]
