# Here we are going to create the Blog Generation task for youtube video
# we will create 2 agents Researcher,content writer
# Researcher will take the transcript from the youtube and give or generate the relevant info
# this relevant info is passed to content writer to generate the blog
from crewai import Agent
from tools import yt_tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile")

# Senior Blog Researcher 
blog_researcher=Agent( 
    role="Blog Researcher from Youtube Videos",
    goal="get the relevant video content for the topic {topic} from Yt Channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI Data Science, MAchine Learning and Gen AI and providing suggesion"
    ),
    tools=[yt_tool],
    llm=model,
    allow_delegation=True     # --------> Transfering ur work to someone else
)

# Senior Blog writer Agent
blog_writer=Agent(
    role="Writer",
    goal="Narrate compelling tech stories about the video {topic} from YT Channel",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft engaging narratives that captivate and educate,bringing new discoveries to light in an accesible manner."
    ),
    tools=[yt_tool],
    llm=model,
    allow_delegation=False
)