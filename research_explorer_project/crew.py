from textwrap import dedent
from typing import Dict
import os

from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ---------- LLM CONFIG ----------
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.4,
    use_native=False,  # avoid requiring crewai[google-genai] native provider
)


class ResearchExplorerCrew:
    """
    Orchestrates agents & tasks for the Research Topic Explorer & Report Generator.
    """

    def __init__(self):
        # 1) Topic Decomposer Agent
        self.topic_decomposer = Agent(
            role="Topic Decomposer",
            goal=(
                "Break a broad research topic into 4–6 meaningful subtopics or questions "
                "that a student should explore."
            ),
            backstory=dedent(
                """
                You are an academic mentor who helps students break down complex
                research topics into smaller, understandable parts.
                """
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        # 2) Info Collector Agent
        self.info_collector = Agent(
            role="Information Collector",
            goal=(
                "For each subtopic, gather clear explanations, key methods, pros/cons."
            ),
            backstory=dedent(
                """
                You are a research assistant for a student and summarize research concepts.
                """
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        # 3) Report Writer Agent
        self.report_writer = Agent(
            role="Report Writer",
            goal=(
                "Combine all collected information into a mini research report."
            ),
            backstory=dedent(
                """
                You write exam-ready mini project reports for B.Tech students.
                """
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        # 4) Presentation Outline Agent
        self.presentation_agent = Agent(
            role="Presentation Designer",
            goal=(
                "Convert the written report into slide-wise bullet points "
                "for a short technical presentation."
            ),
            backstory=dedent(
                """
                You create clean and clear technical PPT outlines.
                """
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        # 5) NEW → Full PPT Content Agent
        self.ppt_content_agent = Agent(
            role="PPT Content Writer",
            goal=(
                "Expand the PPT outline into full slide content for a presentation session."
            ),
            backstory=dedent(
                """
                You create complete slide text based on a PPT outline — including
                slide headings, explanations, bullet points, examples, and notes.
                """
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

    def build_crew(self, topic: str, instructions: str = "") -> Crew:
        extra_note = instructions.strip() or "No extra instructions."

        # Task 1: Topic Decomposition
        topic_task = Task(
            description=dedent(
                f"""
                Break the topic "{topic}" into 4–6 subtopics with explanations.

                # Topic Decomposition
                """
            ),
            expected_output="A list of 4-6 subtopics with a short (1-2 sentence) explanation for each, in markdown.",
            agent=self.topic_decomposer,
        )

        # Task 2: Info collection
        info_task = Task(
            description=dedent(
                """
                Using 'Topic Decomposition', provide definitions, methods, pros/cons.
                # Collected Information
                """
            ),
            expected_output="For each subtopic, provide a definition, key methods, and pros/cons in structured markdown (subheadings).",
            agent=self.info_collector,
            context=[topic_task],
        )

        # Task 3: Report writing
        report_task = Task(
            description=dedent(
                f"""
                Write a complete mini-report for the topic: "{topic}".
                """
            ),
            expected_output="A cohesive mini-report in markdown: introduction, sections for each subtopic, methods, pros/cons, conclusion, and references.",
            agent=self.report_writer,
            context=[topic_task, info_task],
        )

        # Task 4: PPT Outline
        ppt_outline_task = Task(
            description=dedent(
                """
                Convert the full mini-report into a slide-wise outline 
                (8–10 slides) with titles and bullet points.

                Format:
                # Presentation Outline
                ## Slide 1: ...
                - Bullet 1
                - Bullet 2
                """
            ),
            expected_output="An 8-10 slide outline in markdown with slide titles and 3-5 bullet points per slide.",
            agent=self.presentation_agent,
            context=[report_task],
        )

        # ⭐ NEW Task 5: Full PPT Content
        ppt_content_task = Task(
            description=dedent(
                """
                Using the 'Presentation Outline', expand each slide into complete content.

                Format:
                # Full PPT Content

                ## Slide 1: <Title>
                - Full sentences explaining the slide
                - Bullet points with meaningful descriptions
                - Notes for presenter (if needed)

                ## Slide 2: <Title>
                - ...

                Ensure content is clear, student-friendly, and presentation-ready.
                """
            ),
            expected_output="Full slide content for each slide: title, paragraph explanation, bullet details, and optional presenter notes in markdown.",
            agent=self.ppt_content_agent,
            context=[ppt_outline_task, report_task],
        )

        # Build and return the Crew with the defined tasks
        crew = Crew(tasks=[topic_task, info_task, report_task, ppt_outline_task, ppt_content_task])
        return crew

    def run(self, topic: str, instructions: str = "") -> Dict[str, str]:
        crew = self.build_crew(topic, instructions)
        result_markdown = crew.kickoff()
        return {"full_markdown": str(result_markdown)}
