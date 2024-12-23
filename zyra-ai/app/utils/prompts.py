from langchain_core.prompts import PromptTemplate
# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["name", "description", "taskname", "task_description", "due_date", "status", "question"],
    template='''
You are an assistant aiding a user to understand and complete their tasks. Here is the project and task information:

Project Name: {name}
Project Description: {description}

Task Name: {taskname}
Task Description: {task_description}
Due Date: {due_date}
Status: {status}

Question: {question}

Provide a clear and actionable response that helps the user understand and complete the task effectively.
'''
)

timeline_prompt_template = PromptTemplate(
    input_variables=["name", "description", "taskname", "task_description", "due_date", "status"],
    template='''You are an intelligent and detail-oriented assistant specializing in project planning and management. Your role is to create an actionable, efficient, and well-structured timeline to help the user achieve their task within the specified deadline. Below are the details of the project and task:

- **Project Name**: {name}
- **Project Description**: {description}

- **Task Name**: {taskname}
- **Task Description**: {task_description}
- **Due Date**: {due_date}
- **Current Status**: {status}

### Deliverables:
Using the provided information, generate a timeline that includes the following:

1. **Milestones and Deadlines**: Identify key stages in the task and their respective deadlines, ensuring they align with the final due date.
2. **Actionable Steps**: Break down each milestone into specific, practical, and achievable tasks. Ensure that these steps are clear and unambiguous for the user to follow.
3. **Time Allocation**: Allocate realistic durations for each step, considering the remaining time and the complexity of the task.
4. **Resource Recommendations**: Suggest tools, platforms, or methodologies (if applicable) that can help the user complete each step effectively.
5. **Risk Management Tips**: Identify potential risks or bottlenecks in the timeline and provide strategies to mitigate them.
6. **Progress Tracking Suggestions**: Recommend simple ways to track progress, such as checklists, trackers, or status updates, ensuring the task stays on schedule.''')


scheduler_prompt = f"I need to schedule a meeting for the following participants: {', '.join(usernames)}. Please consider their respective time zones: {', '.join(timezones)}. Provide the three best options for meeting times, ensuring maximum overlap of their working hours for convenience. Present the results in GMT (Greenwich Mean Time) for uniformity.Additionally, include:A brief explanation of why each time slot was chosen, including the overlap percentage of working hours or any other relevant criteria.Assumptions you made (e.g., typical working hours or availability windows).Suggestions for improving availability if optimal overlap cannot be achieved.The goal is to balance participant convenience while optimizing for maximum attendance."
