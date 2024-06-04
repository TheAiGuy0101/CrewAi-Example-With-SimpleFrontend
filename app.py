from flask import Flask, render_template, request
import os
import openai
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

app = Flask(__name__)

# Initialize the SerperDevTool (assuming you have the API key set up in your environment)
search_tool = SerperDevTool()

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/kickoff', methods=['POST'])
def kickoff():
    topic = request.form['topic']

    # Create agents
    researcher = Agent(
        role='Senior Researcher',
        goal=f'Uncover groundbreaking technologies in {topic}',
        verbose=True,
        memory=True,
        backstory=(
            "Driven by curiosity, you're at the forefront of"
            "innovation, eager to explore and share knowledge that could change"
            "the world."
        ),
        tools=[search_tool],
        allow_delegation=True
    )

    writer = Agent(
        role='Writer',
        goal=f'Narrate compelling tech stories about {topic}',
        verbose=True,
        memory=True,
        backstory=(
            "With a flair for simplifying complex topics, you craft"
            "engaging narratives that captivate and educate, bringing new"
            "discoveries to light in an accessible manner."
        ),
        tools=[search_tool],
        allow_delegation=False
    )

    # Create tasks
    research_task = Task(
        description=(
            f"Identify the next big trend in {topic}."
            "Focus on identifying pros and cons and the overall narrative."
            "Your final report should clearly articulate the key points"
            "its market opportunities, and potential risks."
        ),
        expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
        tools=[search_tool],
        agent=researcher,
    )

    write_task = Task(
        description=(
            f"Compose an insightful article on {topic}."
            "Focus on the latest trends and how it's impacting the industry."
            "This article should be easy to understand, engaging, and positive."
        ),
        expected_output='A 4 paragraph article on {topic} advancements formatted as markdown.',
        tools=[search_tool],
        agent=writer,
        async_execution=False,
        output_file='new-blog-post.md'
    )

    # Form the crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential
    )

    # Kickoff the crew
    result = crew.kickoff(inputs={'topic': topic})

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
