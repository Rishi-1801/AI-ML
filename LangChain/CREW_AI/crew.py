from crewai import Crew,Process
from agents import blog_writer,blog_researcher
from tasks import research_task,writer_task

# Forming the tech focused crew with some enhanced configuration
crew=Crew(
    agents=[blog_researcher,blog_writer],
    tasks=[research_task,writer_task],
    process=Process.sequential ,# Optionl: Sequentila task exe is default
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

# Start the task execution 
result=crew.kickoff(inputs={"topic":"AI VS ML VS DL VS Data Science"})
print(result)