from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import CSVSearchTool
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama

from wedding_planner.tools.custom_tool import SearchCSV

# import os
# os.environ["OPENAI_API_KEY"] = "NA"

llm = Ollama(
    model = "gemma:2b",
    base_url = "http://localhost:11434")


# country_venue_price_tool = CSVSearchTool()
search_csv = SearchCSV()
#country_venue_price_tool = CSVSearchTool(
#    config=dict(
#        llm=dict(
#            provider="ollama", # or google, openai, anthropic, llama2, ...
#            config=dict(
#                model="gemma:2b",
#                # temperature=0.5,
#                # top_p=1,
#                # stream=true,
#            ),
#        ),
#        embedder=dict(
#            provider="ollama", # or openai, ollama, ...
#            config=dict(
#                model="gemma:2b",
#                # task_type="retrieval_document",
#                # title="Embeddings",
#            ),
#        ),
#    )
#)

@CrewBase
class DestinationWeddingPlannerCrew():
    """Destination Wedding Planner crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def venue_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config['venue_coordinator'],
            tools=[search_csv],
			# llm=llm,
            llm=ChatOpenAI(model_name="gpt-4o", temperature=0.2),
            function_calling_llm=ChatOpenAI(model_name="gpt-4o", temperature=0.2),
            # function_calling_llm=llm,
            verbose=True
        )

    @agent
    def travel_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['travel_agent'],
			llm=llm,
            verbose=True
        )

    @agent
    def event_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['event_planner'],
			llm=llm,
            verbose=True
        )

    @agent
    def catering_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['catering_manager'],
			llm=llm,
            verbose=True
        )

    @agent
    def budget_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['budget_analyst'],
			llm=llm,
            verbose=True
        )

    @task
    def venue_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['venue_research'],
            agent=self.venue_coordinator()
        )

    @task
    def travel_planning_task(self) -> Task:
        return Task(
            config=self.tasks_config['travel_planning'],
            agent=self.travel_agent()
        )

    @task
    def event_timeline_task(self) -> Task:
        return Task(
            config=self.tasks_config['event_timeline'],
            agent=self.event_planner()
        )

    @task
    def menu_design_task(self) -> Task:
        return Task(
            config=self.tasks_config['menu_design'],
            agent=self.catering_manager()
        )

    @task
    def budget_management_task(self) -> Task:
        return Task(
            config=self.tasks_config['budget_management'],
            agent=self.budget_analyst()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Destination Wedding Planner crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=2,
            # process=Process.hierarchical, # Uncomment if you want to use hierarchical process
        )