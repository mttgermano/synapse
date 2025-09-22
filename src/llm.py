from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key="AIzaSyD5rbP-fsG3NB2gQo2tgfDjyRPMRlNpB-0"
)
