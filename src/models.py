from pydantic import BaseModel, Field


class ResponseSchema(BaseModel):
    """Schema for the response from the LLM."""

    thoughts: str = Field(description="The thoughts process to answer the question")
    answer: str = Field(description="The final answer to the question")
