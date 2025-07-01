from langchain_openai import OpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
load_dotenv()

model = OpenAI()
# schema, without annotation
# class Review(TypedDict):
#     summary : str
#     sentiment : str

# schema, with Annotated
# Note:- annotated is used to mention or to provide the detiles about the key
# Note:- Not garanty of data validatio here. Typeddict is for the represntation purpose only.
# class Review(TypedDict):
#     summary : Annotated[str,"Provide summary on the given rivew"]
#     sentiment : Annotated[str,'Provide sentiment like Positive or Nigative only']
#     pros : Annotated[Optional[str],'Provide pros if its list down in the review']
#     cons : Annotated[Optional[str],'Provide cons if its list down in the review']

class Review(TypedDict):
    summary : str = Field(description='Provide summary on the given rivew')
    sentiment : Literal['pos','neg'] = Field(description='Provide sentiment like Positive or Nigative or Neutral')
    pros : Optional[list[str]] = Field(default=None,description='Provide pros if its list down in the review')
    cons : Optional[list[str]] = Field(default=None, description='Provide cons if its list down in the review')
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("The hardware id greate, but software feels bloadted. There are  too many pre-installed apps that I can't remove also the UI looks outdated compare to other brand. Hoping for a software update for this.")

print(result)
print(result['pros'])
print(result['cons'])

