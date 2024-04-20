from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from dotenv import load_dotenv
import os
from langchain_fireworks import ChatFireworks
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain.tools.render import render_text_description
def periodcarerecommender(input_text):
    api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
    wiki=WikipediaQueryRun(api_wrapper=api_wrapper)
    search = TavilySearchAPIWrapper(tavily_api_key=os.getenv("TAVILY_API_KEY"))
    tavily= TavilySearchResults(api_wrapper=search)
    pubmed=PubmedQueryRun()
    tools=[pubmed,wiki,tavily]
    
    load_dotenv()
    
    apikey=os.getenv('FIREWORKS_API_KEY')
   
    llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct", api_key=apikey, max_tokens=300)
    rendered_tools = render_text_description(tools)
   # llm = ChatGoogleGenerativeAI(model="gemini-pro",
   #                          temperature=0.1, safety_settings={
    #    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    #   HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
   #}, google_api_key=GOOGLE_API_KEY)

    prompt_template = f""" Your name is Maitri. You are a medical practitioner and specialize on questions
            regarding female menstruatual health , periods , symptoms related to it , its solutions , 
            diseases related to it and myths related to it.Answer the question as detailed as possible 
            from the given sources, make sure to provide all the details,don't provide the wrong answer to 
            things you do not know and you should not entertain any questions that are not related to female menstruation 
            , periods , symptoms related to it , its solutions , diseases related to it and myths related to it.\n\n 
            Make sure to use only the wiki pubmed tool for information and no other sources strictly.
              Do not provide articles link but you can tell the sources wherever needed.Give a care routine during the phases of menstrual cycle.Strictly  Summarize your answers in 150 tokens.
              Here are the names and descriptions for each tool:

        {rendered_tools}

    """
    # prompt = [ SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=prompt_template)),
    #            MessagesPlaceholder(variable_name='chat_history', optional=True),
    #            HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
    #            MessagesPlaceholder(variable_name='agent_scratchpad')]
   
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompt_template),
        ("user", "{input}")
        
    ]
)
   


    chain = prompt | llm 
    answer=chain.invoke({"input": {input_text}})
    print(answer)
    return(answer.content)
    
    

def main():
    st.set_page_config("Maitri AI-Period Care Recommender")
    st.header("Get Personalized Care Suggestion while exploring your Menstrual Cycle by Maitri AI👩‍⚕️")

    user_phase = st.radio("Select the phase of Menstruation:", ("Menstrual Phase (Day 1 to Day 7)", "Proliferative Phase (Day 8 to Day 11) ", "Ovulation Phase (Day 12 to 17)", "Luteal Phase (Day 18 to Day 28)"))
    if user_phase == "Menstrual Phase (Day 1 to Day 7)":
        user_day = st.number_input("Enter the day of the period phase:", min_value=1, max_value=7)
        abdominal_pain = st.checkbox("Abdominal pain present?")
        period_flow = st.checkbox("Period flow present?")
        if period_flow:
            period_flow_type = st.radio("Select period flow type:", ("Heavy", "Moderate", "Low"))
        
       
        user_question = f"I'm in the Menstrual Phase, Day {user_day}. Abdominal pain: {'Yes' if abdominal_pain else 'No'}, Period flow:{period_flow_type if period_flow else 'None'}. Suggest me how should I take care of myself"
        user_question =user_question + st.text_input("Share if you have any issues that you are facing in the chosen phase of Menstrual Cycle #NoShame ", placeholder="Enter your queries here 🤗 ")
    else:
       
        user_question = st.text_input("Share if you have any issues that you are facing in the chosen phase of Menstrual Cycle #NoShame ", placeholder="Enter your queries here 🤗")
        user_question= f'I am in the {user_phase}. Suggest me how should I take care of myself based on the {user_phase}'+user_question
    
    
    st.sidebar.title("Hey your Personalised Period Guide Assistant is ready")
    st.sidebar.divider()
    st.sidebar.subheader("Ask Personalized Suggestions about issues you are currently facing in your Menstrual Cycle and Maitri AI will help you out !!")
    st.sidebar.divider()
    if st.button("Get Suggestions"):
        st.write(periodcarerecommender(user_question))



if __name__ == "__main__":
    main()
