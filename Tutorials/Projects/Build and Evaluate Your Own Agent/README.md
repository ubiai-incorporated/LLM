# Building an AI Agent Startup Strategist

For the past few years humanity has been at an odd crossroads in tech. Everyone’s talking about “AI this” and “LLM that,” but behind all the buzz, most tools still feel like flashy demos. They answer questions, write summaries, maybe generate some code, and then what? We’re still the one doing the heavy lifting: having to fact check, use tools and put it all together.

But something interesting is happening recently.

A new kind of system is quietly gaining traction. AI agents: A piece of software that can use actual tools to get things done and maintain context over extended interactions.In this tutorial, We’ll walk through a working example called Startup Strategist: an agent designed to help people turn rough ideas into actionable startup plans. 

### You’ll learn:

- How to design an agent’s architecture from the ground up.


- How the agent can make decisions with tools, not just words.


- And how to evaluate whether the agent is doing its job well.


We’ll unpack the entire technical structure, from session based memory and agent design to API tool integration and user experience on Streamlit. 
## The Foundations: What Makes an Effective Strategy Agent?
Before explaining the code, it’s essential to understand the core principles that differentiate a strategy-focused agent from generic AI assistants. Most chatbots excel at retrieving information or answering discrete questions, but they falter when tasked with guiding users through multi-stage, context-dependent workflows. A robust strategy agent must possess three critical attributes:

- **persistent memory:** the ability to retain and reference prior interactions to maintain continuity in decision-making. A founder discussing customer segmentation in one session shouldn’t need to re-explain their entire business model in the next.
- **structured reasoning tools:** that go beyond text generation. In our case, these tools encapsulate startup-specific methodologies, allowing the agent to generate Lean Canvases, design validation experiments, or compare business models systematically.
- **orchestrate workflows:** knowing when to gather information, when to apply analytical frameworks, and when to prompt the user for clarification. This dynamic adaptability is what transforms a collection of functions into a cohesive advisory system.

These principles informed every aspect of our implementation, from the memory architecture to the design of individual tools. The result is an agent that doesn’t just respond to queries but actively structures the entrepreneurial process.

**Let’s start with the implementation**
## The Brain Behind the Operation
At its core, Startup Strategist is a LangChain-powered agent running within a Streamlit interface. It uses tool calling to delegate complex tasks like startup brainstorming or Lean Canvas generation to modular functions, each powered by language model prompts and external APIs.

Our agent uses OpenAI's GPT-4 model as its foundation, but with several crucial enhancements:

```python
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful startup strategist assistant..."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

The temperature setting of 0.7 provides a balance between creativity and focus - essential when brainstorming needs to be both innovative and practical. The prompt template establishes the assistant's personality and capabilities while leaving room for the conversation history and agent's intermediate steps.

## Memory: The Key to Coherent Conversations
A strategist that forgets prior conversations is of little use to an entrepreneur. Our implementation addresses this through a sophisticated memory system that operates at multiple levels. At the most basic level, the agent retains the literal dialogue history, allowing it to reference specific details mentioned earlier. But we’ve augmented this with higher-level context preservation. When a user switches from discussing customer pain points to revenue models, the agent maintains awareness of how these topics interrelate within the startup’s overall context.

In the code this is done by creating a persistent session using ConversationBufferMemory, which allows the agent to maintain long-term understanding throughout the chat:


```python
# Initialize session state with memory
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if 'last_output' not in st.session_state:
    st.session_state.last_output = ""
```
This may seem trivial, but this memory structure is what differentiates an agent from a one-off bot.

## Equipping the Agent: The Tool Ecosystem

The real power of an agent comes from the custom tool it is handed. These tools are  functions with parameters and descriptions that let the agent take action beyond language generation.

In the LangChain framework, this is done using Python decorators, specifically the @tool decorator. When we define a function and mark it with @tool, we’re registering it as a callable action that the agent can invoke during its reasoning process.

Here’s a minimal example:

```python
from langchain.tools import tool

@tool
def say_hello(name: str) -> str:
    """Say hello to a person by name."""
    return f"Hello, {name}!"
```

This function is now a tool. The agent can “see” that it exists, read its description, and choose to use it if the user’s input makes it relevant. For instance, if the user says, “Can you greet Alice?”, the agent will match that request to the say_hello tool and call it with name="Alice".

In the Startup Strategist project, we define several tools that help the agent support the user through key stages of the startup ideation process. Each tool is purpose-built and handles one specific task:

### Brainstorming Engine
This tool uses a custom prompt sent to an external UBIAI model endpoint. It generates actionable startup ideas, breaking down abstract interests into tangible directions. This is particularly useful for non-technical founders who can describe what they care about, but not what business it could turn into.


```python
@tool
def brainstorm_startup_ideas(
    interests: str,
    industry: str = "",
    skills: str = "",
    pain_points: str = ""
) -> dict:
    """Generate creative startup ideas based on interests, industry, skills, and pain points."""
    url = ""
    my_token = ""

    user_prompt = f"""Generate creative and actionable startup or project ideas based on the following:
- Interests: {interests}
- Industry: {industry}
- Skills: {skills}
- Pain Points: {pain_points}
Format the output clearly with bullet points or numbers."""

    data = {
        "input_text": "",
        "system_prompt": "You are a helpful assistant who generates startup ideas tailored to user input.",
        "user_prompt": user_prompt,
        "temperature": 0.7,
        "monitor_model": True,
        "knowledge_base_ids": [],
        "session_id": "",
        "images_urls": []
    }

    try:
        response = requests.post(url + my_token, json=data)
        if response.status_code == 200:
            res = response.json()
            return {"ideas": res.get("output")}
        else:
            return {"error": f"{response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}

```

Notice how we guide the model with specific parameters and a structured prompt. This prevents vague suggestions and ties ideas directly to the user's context.


### Validating Startup Ideas

Many first-time entrepreneurs fall in love with their ideas without validating them. Our validate_idea uses a lean methodology framework and turns a basic description into an interactive validation checklist.

```python

@tool
def validate_idea(
    idea_description: str,
    target_audience: str = "",
    value_proposition: str = "",
    assumptions: str = ""
) -> dict:
    """Validate a startup idea using lean startup framework and provide validation steps."""
    url = ""
    my_token = ""

    user_prompt = f"""Validate the following idea using a lean startup framework. Provide a checklist or set of validation questions that help assess its feasibility.

Idea: {idea_description}
Target Audience: {target_audience}
Value Proposition: {value_proposition}
Key Assumptions: {assumptions}

Give practical and direct validation steps or criteria.
"""

    data = {
        "input_text": "",
        "system_prompt": "You are a lean startup expert helping users validate ideas realistically.",
        "user_prompt": user_prompt,
        "temperature": 0.7,
        "monitor_model": True,
        "knowledge_base_ids": [],
        "session_id": "",
        "images_urls": []
    }

    try:
        response = requests.post(url + my_token, json=data)
        if response.status_code == 200:
            res = response.json()
            return {"validation": res.get("output")}
        else:
            return {"error": f"{response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}
```

You can treat this as the equivalent of hiring a business consultant for the price of an API call.

**Same for the generate lean canvas tool!**

## Backend Utilities: Exporting and Communicating

What makes this agent useful in practice isn't just the ideation, it’s that it completes the loop. Once an idea is structured, users may want to:
Save it as a Word document.


Share it by email.

Both actions are supported as tools.

### Saving to Word

Using python-docx, the system formats output with headers and bullet points, then saves it locally:


```python
@tool
def save_to_word_doc(content: str, filename: str = "business_canvas.docx") -> dict:
    """Save text content to a Word document."""
    try:
        doc = Document()
        doc.add_heading("Business Canvas Document", 0)
        
        # Use the last output if no specific content provided
        if not content and st.session_state.last_output:
            content = st.session_state.last_output
            
        for line in content.split('\n'):
            doc.add_paragraph(line)
        doc.save(filename)
        return {"filename": filename, "message": f"Document saved as {filename}"}
    except Exception as e:
        return {"error": str(e)}
```

### Email Integration
Business founders may want to send their canvas to a mentor, investor, or teammate. The send_email() tool uses SMTP over TLS, configured to work with Gmail or others:

```python
@tool
def send_email(
    recipient_email: str,
    subject: str,
    body: str,
    sender_email: str = None,
    sender_password: str = None
) -> dict:
    """Send an email using SMTP."""
    try:
        from_email = sender_email if sender_email else EMAIL_CONFIG["sender_email"]
        password = sender_password if sender_password else EMAIL_CONFIG["sender_password"]

        if not from_email or not password:
            return {"error": "Email credentials not configured"}

        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, recipient_email, msg.as_string())
        server.quit()

        return {"status": "success", "message": f"Email sent to {recipient_email}"}
    except Exception as e:
        return {"error": str(e)}
```


With minimal code, the agent becomes not just a thinking partner, but a working assistant.

## Registering Tools with the Agent
Once the tools are defined, we pass them into the agent during initialization. This happens in the initialize_agent_executor() function:


```python
tools = [
    brainstorm_startup_ideas,
    validate_idea,
    generate_lean_canvas,
    save_to_word_doc,
    send_email
]
```


We then provide this list to LangChain’s initialize_agent() method:


```python
return initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=st.session_state.memory,
    prompt=prompt
)
```

With this setup, the agent is no longer just generating text — it’s choosing actions from a defined toolbox, passing arguments to those functions, and returning structured outputs
Prompt Engineering with LangChain
LangChain’s initialize_agent() function is where everything comes together. We define the logic using ChatPromptTemplate, add a MessagesPlaceholder for history, and let the agent reason which tool to use.


```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "...strategist assistant..."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

This prompt setup is crucial: it frames the role of the agent, reinforces behavioral rules, and makes sure it responds with relevance.
Finally, the agent is created with:


```python
initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, memory=..., prompt=...)
```

This is the beating heart of our application.
Here is a quick look at the application:

![image](https://github.com/user-attachments/assets/f5ab7dfd-1266-418b-9060-4c46441db13a)

![image](https://github.com/user-attachments/assets/86831db1-42f9-4d6d-9bf6-d43888221763)


## The Critical Need for Agent evaluation

Building an agent that works is one thing. Knowing whether it’s working well, that’s a different challenge entirely. It’s easy to get excited when an agent completes a task or calls the right function. **But is it doing the right task?** Did it choose the best tool for the input? Was the response helpful, or just verbose? These aren’t binary outcomes. They require judgment. And in systems that blend reasoning, tool usage, and memory, quality is hard to pin down without a structured evaluation process.

This is where most developer workflows still fall short. We write agents, we test them ad hoc, and we adjust based on feel. But if we want to build agents we can trust in production, especially ones that make decisions on behalf of users, we need to treat evaluation as a first-class concern, not an afterthought.

In the second part of this tutorial, we’ll explore what that looks like.
We’ll walk through how to evaluate an agent’s behavior at the decision-making level using a technique called **LLM-as-a-Judge**. The idea is simple: we give a separate, neutral language model a transcript and ask it to grade the agent’s decisions, explanations, or tool usage. This lets us move beyond "did it run?" into "did it help?", and it allows us to run systematic comparisons between prompts, memory strategies, or tool designs.

**Open the Agent_Evaluation.Ipynb to start learning!**

